"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *

import argparse
import wandb

torch.manual_seed(0)
np.random.seed(0)

################################################################
# thermodynamic collision
################################################################
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

def KINET_DSMC2(x, v, a, rand_vec, training=False):
    dt = 1
    coll_coef = 0.9

    bs, chnl_old, n_old = x.shape
    _, chnl, _ = rand_vec.shape
    device = x.device 
    n_divide = chnl / chnl_old
    n = int(n_old / n_divide)
    a = a.reshape(bs, chnl, n)
    v = v.reshape(bs, chnl, n)
    x = x.reshape(bs, chnl, n)

    # v = v + a * dt
    v = a * dt

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
    v_r_max = v_r_max.view(bs, 1, 1)

    mask = v_r / v_r_max * u_x
    # batchnorm_mask = nn.BatchNorm1d(n).to(device)
    # mask = batchnorm_mask(mask)

    if not training:
        coll_coef = 0

    collision_mask = mask > (1 -  coll_coef) # (bs, n, n), mask of particles that collide, equivalent to bernoulli(p=v_r/v_max*u_x)
        
    delta_v = torch.zeros((bs, chnl, n, n))

    v_r_leaving = project_to_antisymmetric(v_r, chnl, rand_vec)

    delta_v = v_cm + v_r_leaving - v.unsqueeze(-1)

    v = v + torch.sum(delta_v * collision_mask.unsqueeze(1), dim=2)

    collision_mask = collision_mask.float()
    # print(collision_mask.shape)
    eye_matrix = torch.eye(n, device=collision_mask.device).unsqueeze(0)  # shape: (1, n, n)
    eye_matrix = eye_matrix.expand(bs, -1, -1)  # shape: (bs, n, n)
    collision_mask = collision_mask + eye_matrix
    # print(torch.sum(x_cm * collision_mask.unsqueeze(1), dim=2).shape)
    # print(torch.sum(collision_mask, dim=2).unsqueeze(1).shape)
    x = torch.sum(x_cm * collision_mask.unsqueeze(1), dim=2) / torch.sum(collision_mask, dim=2).unsqueeze(1)
    x = x + v * dt

    return x.view(bs, int(chnl/n_divide), n_old), v.view(bs, int(chnl/n_divide), n_old)

################################################################
#  1d fourier layer
################################################################

def compl_mul1d(a, b):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    op = partial(torch.einsum, "bix,iox->box")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, 2))

    def forward(self, x):
        batchsize = x.shape[0] # x: [batchsize, resolution, in_channels]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft_complex = torch.fft.rfft(x, dim=2, norm="ortho")
        x_ft = torch.view_as_real(x_ft_complex)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1] = compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        out_ft_complex = torch.view_as_complex(out_ft)
        x = torch.fft.irfft(out_ft_complex, n=x.size(-1), dim=2, norm="ortho")
        return x

class SimpleBlock1d(nn.Module):
    def __init__(self, modes, width):
        super(SimpleBlock1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):

        x = self.fc0(x)
        x = x.permute(0, 2, 1) # (bs, width, s)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class SimpleBlock1d_kinet2(nn.Module):
    def __init__(self, modes, width):
        super(SimpleBlock1d_kinet2, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, v=None, rand_vec=None):

        x = self.fc0(x)
        x = x.permute(0, 2, 1) # (bs, width, s)

        a = self.conv0(x) # acceleration
        p = self.w0(x) # position
        vec = rand_vec()
        x, v = KINET_DSMC2(p, v, a, vec, self.training)
        x = F.relu(x)
        v = F.relu(v)

        a = self.conv1(x)
        p = self.w1(x)
        vec = rand_vec()
        x, v = KINET_DSMC2(p, v, a, vec, self.training)
        x = F.relu(x)
        v = F.relu(v)

        a = self.conv2(x)
        p = self.w2(x)
        vec = rand_vec()
        x, v = KINET_DSMC2(p, v, a, vec, self.training)
        x = F.relu(x)
        v = F.relu(v)

        a = self.conv3(x)
        p = self.w3(x)
        vec = rand_vec()
        x, v = KINET_DSMC2(p, v, a, vec, self.training)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class Net1d(nn.Module):
    def __init__(self, modes, width):
        super(Net1d, self).__init__()
        """
        A wrapper function
        """
        self.conv1 = SimpleBlock1d(modes, width)


    def forward(self, x, v=None, rand_vec=None):
        x = self.conv1(x)
        return x.squeeze()

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c

class Net1d_kinet2(nn.Module):
    def __init__(self, modes, width):
        super(Net1d_kinet2, self).__init__()
        """
        A wrapper function
        """
        self.conv1 = SimpleBlock1d_kinet2(modes, width)


    def forward(self, x, v=None, rand_vec=None):
        x = self.conv1(x, v, rand_vec)
        return x.squeeze()

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c


################################################################
#  configurations
################################################################
parser = argparse.ArgumentParser(description='PyTorch FNO 1D')
parser.add_argument('--ntrain', type=int, default=1000, help='Number of training samples')
parser.add_argument('--ntest', type=int, default=100, help='Number of testing samples')
parser.add_argument('--sub', type=int, default=2**3, help='Subsampling rate')
parser.add_argument('--n_divide', type=int, default=4, help='Dimension of the collision')
parser.add_argument('--s', type=int, default=2**10, help='resolution')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
parser.add_argument('--step_size', type=int, default=100, help='Step size for learning rate decay')
parser.add_argument('--gamma', type=float, default=0.5, help='Gamma for learning rate decay')
parser.add_argument('--modes', type=int, default=16, help='Number of Fourier modes')
parser.add_argument('--width', type=int, default=64, help='Width of the network')
parser.add_argument('--model_name', type=str, default='FNO1d', help='FNO1d or FNO1d_kinet2')
args = parser.parse_args()

device = f'cuda:0'
if_wandb = True

if if_wandb:
    wandb.init(project="kinet-burgers", config=args, name=f"{args.model_name}")
################################################################
# read data
################################################################

# Data is of the shape (number of samples, grid size)
dataloader = MatReader('data/burgers_data_R10.mat')
x_data = dataloader.read_field('a')[:,::args.sub]
y_data = dataloader.read_field('u')[:,::args.sub]

x_train = x_data[:args.ntrain,:]
y_train = y_data[:args.ntrain,:]
x_test = x_data[-args.ntest:,:]
y_test = y_data[-args.ntest:,:]

# cat the locations information
grid = np.linspace(0, 2*np.pi, args.s).reshape(1, args.s, 1)
grid = torch.tensor(grid, dtype=torch.float)
x_train = torch.cat([x_train.reshape(args.ntrain,args.s,1), grid.repeat(args.ntrain,1,1)], dim=2)
x_test = torch.cat([x_test.reshape(args.ntest,args.s,1), grid.repeat(args.ntest,1,1)], dim=2)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

# model
if args.model_name == 'FNO1d':
    model = Net1d(args.modes, args.width)
    rand_vec = None
    v = None
elif args.model_name == 'FNO1d_kinet2':
    model = Net1d_kinet2(args.modes, args.width)
    rand_vec = rand_vec(args.batch_size, args.width * args.n_divide, args.s / args.n_divide, device)
    v = torch.zeros((args.batch_size, args.width, args.s)).to(device)

model = model.to(device)
print('model params: ', model.count_params())


################################################################
# training and evaluation
################################################################
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

myloss = LpLoss(size_average=False)
for ep in range(args.epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x, v, rand_vec)

        mse = F.mse_loss(out, y, reduction='mean')
        # mse.backward()
        l2 = myloss(out.view(args.batch_size, -1), y.view(args.batch_size, -1))
        l2.backward() # use the l2 relative loss

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x, v, rand_vec)
            test_l2 += myloss(out.view(args.batch_size, -1), y.view(args.batch_size, -1)).item()

    train_mse /= len(train_loader)
    train_l2 /= args.ntrain
    test_l2 /= args.ntest

    t2 = default_timer()
    print('epoch:', ep, 'time: %.2f' %(t2-t1), train_mse, 'train l2: %.8f'%(train_l2), 'test l2: %.8f' %(test_l2))
    if if_wandb:
        wandb.log({"train_loss": train_mse, "train_l2": train_l2, "test_l2": test_l2})

# torch.save(model, 'model/ns_fourier_burgers_8192')
# pred = torch.zeros(y_test.shape)
# index = 0
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
# with torch.no_grad():
#     for x, y in test_loader:
#         test_l2 = 0
#         x, y = x.to(device), y.to(device)

#         out = model(x)
#         pred[index] = out

#         test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
#         print(index, test_l2)
#         index = index + 1

# scipy.io.savemat('pred/burger_test.mat', mdict={'pred': pred.cpu().numpy()})
