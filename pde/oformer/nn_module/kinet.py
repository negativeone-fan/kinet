import torch
import torch.nn as nn

class rand_vec(nn.Module):
    def __init__(self, bs, chnl, n, device=None):
        super(rand_vec, self).__init__()
        self.bs = bs
        self.chnl = chnl
        self.n = n
        self.rand_vec = torch.randn(self.bs, self.chnl, int(self.n*(self.n-1)/2)).to(device)
        self.rand_ori = self.rand_vec / torch.norm(self.rand_vec, dim=1, keepdim=True)
        self.device = device
    
    def forward(self):
        x = -4 + (-7 + 4) * torch.rand(1, device=self.device)
        sign = torch.randint(0, 2, (1,), device=self.device) * 2 - 1
        self.rand_vec = self.rand_vec * self.rand_ori + self.rand_ori + sign * torch.exp(x)
        self.rand_vec = self.rand_vec / torch.norm(self.rand_vec, dim=1, keepdim=True)
        return self.rand_vec

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

def KINET_DSMC(x, v, rand_vec, training=False):
    dt = 1
    coll_coef = 0.9

    bs, n_token, width = x.shape
    device = x.device 
    _, chnl, _ = rand_vec.shape
    rand_vec = rand_vec.to(device)
    new_bs = bs * n_token
    n = int(width / chnl)

    v = v.reshape(new_bs, chnl, n)
    x = x.reshape(new_bs, chnl, n)

    # v = v + a * dt
    # v = a * dt

    x_r = x.unsqueeze(-1) - x.unsqueeze(-2) # (bs, chnl, n_particles, n_particles)
    v_r = v.unsqueeze(-1) - v.unsqueeze(-2) # (bs, chnl, n_particles, n_particles)
    v_cm = (v.unsqueeze(-1) + v.unsqueeze(-2)) / 2 # (bs, chnl, n_particles, n_particles)
    x_cm = (x.unsqueeze(-1) + x.unsqueeze(-2)) / 2 # (bs, chnl, n_particles, n_particles)

    x_r = torch.norm(x_r, dim=1)  # (bs, n, n), distance matrix
    v_r = torch.norm(v_r, dim=1)  # (bs, n, n), relative velocity matrix
    u_x = torch.exp(-x_r)  # distance potential, max=1, min=0

    # for each bs, find the maximum relative velocity
    v_r_max, _ = v_r.max(dim=1, keepdim=False)  # (bs, n)
    v_r_max, _ = v_r_max.max(dim=1, keepdim=False)  # (bs)
    v_r_max = v_r_max.view(new_bs, 1, 1)

    mask = v_r / v_r_max * u_x
    # batchnorm_mask = nn.BatchNorm1d(n).to(device)
    # mask = batchnorm_mask(mask)

    if not training:
        coll_coef = 0

    collision_mask = mask > (1 -  coll_coef) # (bs, n, n), mask of particles that collide, equivalent to bernoulli(p=v_r/v_max*u_x)
        
    delta_v = torch.zeros((new_bs, chnl, n, n))

    v_r_leaving = project_to_antisymmetric(v_r, chnl, rand_vec)

    delta_v = v_cm + v_r_leaving - v.unsqueeze(-1)

    v = v + torch.sum(delta_v * collision_mask.unsqueeze(1), dim=2)

    collision_mask = collision_mask.float()
    # print(collision_mask.shape)
    eye_matrix = torch.eye(n, device=collision_mask.device).unsqueeze(0)  # shape: (1, n, n)
    eye_matrix = eye_matrix.expand(new_bs, -1, -1)  # shape: (bs, n, n)
    collision_mask = collision_mask + eye_matrix
    # print(torch.sum(x_cm * collision_mask.unsqueeze(1), dim=2).shape)
    # print(torch.sum(collision_mask, dim=2).unsqueeze(1).shape)
    x = torch.sum(x_cm * collision_mask.unsqueeze(1), dim=2) / torch.sum(collision_mask, dim=2).unsqueeze(1)
    x = x + v * dt

    return x.view(bs, n_token, width)