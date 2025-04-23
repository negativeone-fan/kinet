import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block[0], 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block[1], 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block[2], 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block[3], 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block[3].expansion, num_classes)

        self.velocity = torch.randn(128, 64, 32, 32)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, rand_ver_1, rand_ver_2, rand_vec_3, rand_vec_4):
        position = F.relu(self.bn1(self.conv1(x))) # (bs, 3, 32, 32) -> (bs, 64, 32, 32)
        bs, _, _, _ = position.shape
        velocity = self.velocity[:bs, :, :, :]
        # print(position.shape, velocity.shape, rand_vec.shape)
        (position, velocity, _) = self.layer1((position, velocity, rand_ver_1)) # for resnet18, (bs, 64, 32, 32) -> (bs, 64, 32, 32)
        (position, velocity, _) = self.layer2((position, velocity, rand_ver_2)) # for resnet18, (bs, 64, 32, 32) -> (bs, 128, 16, 16)
        (position, velocity, _) = self.layer3((position, velocity, rand_vec_3)) # for resnet18, (bs, 128, 16, 16) -> (bs, 256, 8, 8)
        (position, velocity, _) = self.layer4((position, velocity, rand_vec_4)) # for resnet18, (bs, 256, 8, 8) -> (bs, 512, 4, 4)
        out = F.avg_pool2d(position, 4) # (bs, 512, 4, 4) -> (bs, 512, 1, 1)
        out = out.view(out.size(0), -1) # (bs, 512, 1, 1) -> (bs, 512)
        out = self.linear(out) # (bs, 512) -> (bs, 100)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, inputs):
        (x, v, rand_vec) = inputs
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        v = out
        out += self.shortcut(x)
        out = F.relu(out)
        return (out, v, rand_vec)

def project_to_antisymmetric(v_r, chnl, rand_vec):
    """
    Projects the independent elements (upper triangle) of the symmetric matrix v_r (bs, n, n)
    using a random unit vector from a chnl-dimensional spherical distribution, and constructs
    an antisymmetric matrix v_r_leaving (bs, chnl, n, n) with:
      - Diagonal elements set to 0;
      - For i < j, v_r_leaving[:, :, i, j] = v_r[:, i, j] * (random unit vector)
      - For i > j, v_r_leaving[:, :, i, j] = -v_r_leaving[:, :, j, i]
    """
    bs, n, _ = v_r.shape
    device = v_r.device

    # Get the indices of the upper triangle (excluding the diagonal), total of num_pairs = n(n-1)/2 elements
    indices = torch.triu_indices(n, n, offset=1)  # shape: [2, num_pairs]

    # Extract the independent upper-triangle elements from v_r
    v_r_upper = v_r[:, indices[0], indices[1]] # (bs, num_pairs)

    # Project the upper-triangle scalar to the chnl-dimensional space
    # Expand v_r_upper to shape (bs, 1, num_pairs) and multiply element-wise with the random unit vectors
    rand_vec = rand_vec[:bs, :, :]
    proj_upper = v_r_upper.unsqueeze(1) * rand_vec  # shape: (bs, chnl, num_pairs)

    # Initialize the output tensor with shape (bs, chnl, n, n)
    v_r_leaving = torch.zeros(bs, chnl, n, n, device=device)

    # Fill in the upper triangle (i < j) with the projected results
    v_r_leaving[:, :, indices[0], indices[1]] = proj_upper

    # Set the lower triangle (i > j) as the negative of the corresponding upper triangle
    v_r_leaving[:, :, indices[1], indices[0]] = -proj_upper

    return v_r_leaving

class KINET_DSMC(nn.Module): # by fyf, 改变了发生碰撞的参考系, 将残差链接的输入由v改为a, 将碰撞过程视作n个chl维粒子的碰撞
    ''' Kinetic Monte Carlo Simulation in channel dimension
    Reference https://medium.com/swlh/create-your-own-direct-simulation-monte-carlo-with-python-3b9f26fa05ab
    Reference https://github.com/pmocz/dsmc-python/blob/main/dsmc.py
    '''
    def __init__(self):
        super().__init__()
        self.dt = 1 # time step
        self.L = 1 # length of hypercube [-L, L]^d
        self.v_r_max = 10 # over-estimated maximum relative velocity
        self.coll_coef = 0.5 # collision coefficient controls the number of max collisions

    def forward(self, x, v, a, rand_vec):
        '''
        Input:
            x, v, a: position, velocity and acceleration of particles, shape (B, C, X, Y) 
        Output:
            x, v: updated position and velocity of particles, shape (B, C, X, Y) 
        '''

        bs, chnl_old, h, w = x.shape
        _, chnl, _ = rand_vec.shape
        device = x.device 
        n = h * w # number of particles

        n_divide = chnl / chnl_old
        n = int(n / n_divide)
        
        a = a.view(bs, chnl, n)
        v = v.view(bs, chnl, n)
        x = x.view(bs, chnl, n)

        # batchnorm_a = nn.BatchNorm1d(chnl).to(device)
        # batchnorm_v = nn.BatchNorm1d(chnl).to(device)
        # batchnorm_x = nn.BatchNorm1d(chnl).to(device)
        # a = batchnorm_a(a)
        # v = batchnorm_v(v)
        # x = batchnorm_x(x)

        v = v + a * self.dt
        # v = a * self.dt
        
        x_r = x.unsqueeze(-1) - x.unsqueeze(-2) # (bs, chnl, n_particles, n_particles)
        v_r = v.unsqueeze(-1) - v.unsqueeze(-2) # (bs, chnl, n_particles, n_particles)
        v_cm = (v.unsqueeze(-1) + v.unsqueeze(-2)) / 2 # (bs, chnl, n_particles, n_particles)

        x_r = torch.norm(x_r, dim=1)  # (bs, n, n), distance matrix
        v_r = torch.norm(v_r, dim=1)  # (bs, n, n), relative velocity matrix
        u_x = torch.exp(-x_r)  # distance potential, max=1, min=0

        # for each bs, find the maximum relative velocity
        v_r_max, _ = v_r.max(dim=1, keepdim=False)  # (bs, n)
        v_r_max, _ = v_r_max.max(dim=1, keepdim=False)  # (bs)
        v_r_max = v_r_max.view(bs, 1, 1)

        mask = v_r / v_r_max * u_x
        # batchnorm_mask = nn.BatchNorm1d(n).to(device)
        # mask = batchnorm_mask(mask)

        if self.training:
            coll_coef = self.coll_coef
        else:
            coll_coef = 0

        collision_mask = mask > (1 -  coll_coef) # (bs, n, n), mask of particles that collide, equivalent to bernoulli(p=v_r/v_max*u_x)
        
        delta_v = torch.zeros((bs, chnl, n, n))

        v_r_leaving = project_to_antisymmetric(v_r, chnl, rand_vec)

        delta_v = v_cm + v_r_leaving - v.unsqueeze(-1)

        v = v + torch.sum(delta_v * collision_mask.unsqueeze(1), dim=2)

        x = x + v * self.dt

        # # 2. Wall collisions
        # # trace the straight-line trajectory to the top wall, bounce it back
        # hit_wall = x.abs() > self.L
        # dt = (self.L - x.abs()[hit_wall]) / v[hit_wall] # time after collision
        # v[hit_wall] = -v[hit_wall]  # reverse velocity
        # dx = torch.zeros_like(x)
        # dx[hit_wall] = 2 * v[hit_wall] * dt # one dt = moving back to wall, another dt = bouncing.
        # x = x + dx

        return x.view(bs, int(chnl/n_divide), h, w), v.view(bs, int(chnl/n_divide), h, w)
    
class KINET_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(KINET_BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.kinet_dsmc = KINET_DSMC()

    def forward(self, inputs):
        (x, v, rand_vec) = inputs
        a = F.relu(self.bn1(self.conv1(x)))
        a = self.bn2(self.conv2(a))
        out, v = self.kinet_dsmc(self.shortcut(x), self.shortcut(v), a, rand_vec) # perform collision
        out = F.relu(out)
        return (out, v, rand_vec)
    
def ResNet18():
    return ResNet([BasicBlock, BasicBlock, BasicBlock, BasicBlock], [2, 2, 2, 2])

def KINET_ResNet18():
    return ResNet([BasicBlock, BasicBlock, BasicBlock, KINET_BasicBlock], [2, 2, 2, 2])

def KINET_ResNet18_2():
    return ResNet([BasicBlock, BasicBlock, KINET_BasicBlock, KINET_BasicBlock], [2, 2, 2, 2])

def KINET_ResNet18_3():
    return ResNet([BasicBlock, KINET_BasicBlock, KINET_BasicBlock, KINET_BasicBlock], [2, 2, 2, 2])

def KINET_ResNet18_4():
    return ResNet([BasicBlock, KINET_BasicBlock, KINET_BasicBlock, KINET_BasicBlock], [2, 2, 2, 2])