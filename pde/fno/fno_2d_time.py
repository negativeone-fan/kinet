"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which uses a recurrent structure to propagates in time.
"""


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities3 import *

# import operator
# from functools import reduce
# from functools import partial

from timeit import default_timer

import copy
import os
import argparse
import wandb

from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)

################################################################
# parameters plot
################################################################
def mkdirs(fn):  # Create directorys
    if not os.path.isdir(fn):
        os.makedirs(fn)
    return fn

def get_parameter(checkpoint):
    conv0_weight1 = checkpoint['conv0.weights1']
    conv0_weight2 = checkpoint['conv0.weights2']
    conv1_weight1 = checkpoint['conv1.weights1']
    conv1_weight2 = checkpoint['conv1.weights2']
    conv2_weight1 = checkpoint['conv2.weights1']
    conv2_weight2 = checkpoint['conv2.weights2']
    conv3_weight1 = checkpoint['conv3.weights1']
    conv3_weight2 = checkpoint['conv3.weights2']
    w0_weight = checkpoint['w0.weight']
    w0_bias = checkpoint['w0.bias']
    w1_weight = checkpoint['w1.weight']
    w1_bias = checkpoint['w1.bias']
    w2_weight = checkpoint['w2.weight']
    w2_bias = checkpoint['w2.bias']
    w3_weight = checkpoint['w3.weight']
    w3_bias = checkpoint['w3.bias']
    return conv0_weight1, conv0_weight2, conv1_weight1, conv1_weight2, conv2_weight1, conv2_weight2, conv3_weight1, conv3_weight2, w0_weight, w0_bias, w1_weight, w1_bias, w2_weight, w2_bias, w3_weight, w3_bias

def normalize_vectorgroup(checkpoint):
    conv0_weight1, conv0_weight2, conv1_weight1, conv1_weight2, conv2_weight1, conv2_weight2, conv3_weight1, conv3_weight2, w0_weight, w0_bias, w1_weight, w1_bias, w2_weight, w2_bias, w3_weight, w3_bias = get_parameter(checkpoint)
    conv0_weight1 = conv0_weight1.reshape(conv0_weight1.shape[0] * conv0_weight1.shape[1], conv0_weight1.shape[2] * conv0_weight1.shape[3]) # for conv_weight: [width, width, modes1, modes2] -> [width * width, modes1 * modes2]
    conv0_weight2 = conv0_weight2.reshape(conv0_weight2.shape[0] * conv0_weight2.shape[1], conv0_weight2.shape[2] * conv0_weight2.shape[3])
    conv1_weight1 = conv1_weight1.reshape(conv1_weight1.shape[0] * conv1_weight1.shape[1], conv1_weight1.shape[2] * conv1_weight1.shape[3])
    conv1_weight2 = conv1_weight2.reshape(conv1_weight2.shape[0] * conv1_weight2.shape[1], conv1_weight2.shape[2] * conv1_weight2.shape[3])
    conv2_weight1 = conv2_weight1.reshape(conv2_weight1.shape[0] * conv2_weight1.shape[1], conv2_weight1.shape[2] * conv2_weight1.shape[3])
    conv2_weight2 = conv2_weight2.reshape(conv2_weight2.shape[0] * conv2_weight2.shape[1], conv2_weight2.shape[2] * conv2_weight2.shape[3])
    conv3_weight1 = conv3_weight1.reshape(conv3_weight1.shape[0] * conv3_weight1.shape[1], conv3_weight1.shape[2] * conv3_weight1.shape[3])
    conv3_weight2 = conv3_weight2.reshape(conv3_weight2.shape[0] * conv3_weight2.shape[1], conv3_weight2.shape[2] * conv3_weight2.shape[3])
    w0_weight = w0_weight.squeeze(-1).squeeze(-1) # for w_weight: [width, width, 1, 1] -> [width, width]
    w0_bias = torch.unsqueeze(w0_bias,dim=1) # for w_bias: [width] -> [width, 1]
    w1_weight = w1_weight.squeeze(-1).squeeze(-1)
    w1_bias = torch.unsqueeze(w1_bias,dim=1)
    w2_weight = w2_weight.squeeze(-1).squeeze(-1)
    w2_bias = torch.unsqueeze(w2_bias,dim=1)
    w3_weight = w3_weight.squeeze(-1).squeeze(-1)
    w3_bias = torch.unsqueeze(w3_bias,dim=1)
    conv0 = torch.cat((conv0_weight1,conv0_weight2),dim=1) # [width * width, 2 * (modes1 * modes2)]
    conv1 = torch.cat((conv1_weight1,conv1_weight2),dim=1)
    conv2 = torch.cat((conv2_weight1,conv2_weight2),dim=1)
    conv3 = torch.cat((conv3_weight1,conv3_weight2),dim=1)
    w0 = torch.cat((w0_weight,w0_bias),dim=1) # [width, width+1]
    w1 = torch.cat((w1_weight,w1_bias),dim=1)
    w2 = torch.cat((w2_weight,w2_bias),dim=1)
    w3 = torch.cat((w3_weight,w3_bias),dim=1)
    conv0 = conv0.detach().cpu().numpy()
    conv1 = conv1.detach().cpu().numpy()
    conv2 = conv2.detach().cpu().numpy()
    conv3 = conv3.detach().cpu().numpy()
    w0 = w0.detach().cpu().numpy()
    w1 = w1.detach().cpu().numpy()
    w2 = w2.detach().cpu().numpy()
    w3 = w3.detach().cpu().numpy()
    conv0_norm = np.linalg.norm(conv0,axis=1)
    conv1_norm = np.linalg.norm(conv1,axis=1)
    conv2_norm = np.linalg.norm(conv2,axis=1)
    conv3_norm = np.linalg.norm(conv3,axis=1)
    w0_norm = np.linalg.norm(w0,axis=1)
    w1_norm = np.linalg.norm(w1,axis=1)
    w2_norm = np.linalg.norm(w2,axis=1)
    w3_norm = np.linalg.norm(w3,axis=1)
    conv0_mask = conv0_norm > 0
    conv1_mask = conv1_norm > 0
    conv2_mask = conv2_norm > 0
    conv3_mask = conv3_norm > 0
    w0_mask = w0_norm > 0
    w1_mask = w1_norm > 0
    w2_mask = w2_norm > 0
    w3_mask = w3_norm > 0
    conv0 = conv0[conv0_mask]
    conv1 = conv1[conv1_mask]
    conv2 = conv2[conv2_mask]
    conv3 = conv3[conv3_mask]
    w0 = w0[w0_mask]
    w1 = w1[w1_mask]
    w2 = w2[w2_mask]
    w3 = w3[w3_mask]
    conv0_norm = conv0_norm[conv0_mask]
    conv1_norm = conv1_norm[conv1_mask]
    conv2_norm = conv2_norm[conv2_mask]
    conv3_norm = conv3_norm[conv3_mask]
    w0_norm = w0_norm[w0_mask]
    w1_norm = w1_norm[w1_mask]
    w2_norm = w2_norm[w2_mask]
    w3_norm = w3_norm[w3_mask]
    conv0 = conv0 / conv0_norm[:,np.newaxis]
    conv1 = conv1 / conv1_norm[:,np.newaxis]
    conv2 = conv2 / conv2_norm[:,np.newaxis]
    conv3 = conv3 / conv3_norm[:,np.newaxis]
    w0 = w0 / w0_norm[:,np.newaxis]
    w1 = w1 / w1_norm[:,np.newaxis]
    w2 = w2 / w2_norm[:,np.newaxis]
    w3 = w3 / w3_norm[:,np.newaxis]
    return conv0, conv1, conv2, conv3, w0, w1, w2, w3


def seperate_vectors_by_eigenvector(vector_group):
    mask = np.linalg.norm(vector_group,axis=1) > 0
    vector_group = vector_group[mask]
    similarity_matrix = np.dot(vector_group,vector_group.transpose())
    w,v = np.linalg.eig(similarity_matrix)  #eigenvalues (M,) and eigenvectors (M,M)
    index = np.argmax(w) #find the largest eigenvalue index
    tmpeig = v[:,index] #find the corresponding eigenvector of largest eigenvalue
    order_mask = np.argsort(tmpeig) #sort the eigenvector
    
    similarity_matrix = similarity_matrix[order_mask,:] #sort the similarity matrix by the largest eigenvector
    similarity_matrix = similarity_matrix[:,order_mask]
    return similarity_matrix

def plot_weight_heatmap_eigen(weight, path, nota=''):
    conv0, conv1, conv2, conv3, w0, w1, w2, w3 = normalize_vectorgroup(weight)

    conv0_similarity_matrix = seperate_vectors_by_eigenvector(conv0)
    fn = mkdirs(os.path.join('%s'%path,'conv0_similarity_kinet'))
    plt.figure()
    plt.pcolormesh(np.abs(conv0_similarity_matrix),vmin=-1,vmax=1,cmap='YlGnBu')
    plt.colorbar()
    plt.xlabel('index',fontsize=18)
    plt.xticks(fontsize=18)
    plt.ylabel('index',fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(fn,'%s'%nota))
    plt.close()

    conv1_similarity_matrix = seperate_vectors_by_eigenvector(conv1)
    fn = mkdirs(os.path.join('%s'%path,'conv1_similarity_kinet'))
    plt.figure()
    plt.pcolormesh(np.abs(conv1_similarity_matrix),vmin=-1,vmax=1,cmap='YlGnBu')
    plt.colorbar()
    plt.xlabel('index',fontsize=18)
    plt.xticks(fontsize=18)
    plt.ylabel('index',fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(fn,'%s'%nota))
    plt.close()

    conv2_similarity_matrix = seperate_vectors_by_eigenvector(conv2)
    fn = mkdirs(os.path.join('%s'%path,'conv2_similarity_kinet'))
    plt.figure()
    plt.pcolormesh(np.abs(conv2_similarity_matrix),vmin=-1,vmax=1,cmap='YlGnBu')
    plt.colorbar()
    plt.xlabel('index',fontsize=18)
    plt.xticks(fontsize=18)
    plt.ylabel('index',fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(fn,'%s'%nota))
    plt.close()

    conv3_similarity_matrix = seperate_vectors_by_eigenvector(conv3)
    fn = mkdirs(os.path.join('%s'%path,'conv3_similarity_kinet'))
    plt.figure()
    plt.pcolormesh(np.abs(conv3_similarity_matrix),vmin=-1,vmax=1,cmap='YlGnBu')
    plt.colorbar()
    plt.xlabel('index',fontsize=18)
    plt.xticks(fontsize=18)
    plt.ylabel('index',fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(fn,'%s'%nota))
    plt.close()

    w0_similarity_matrix = seperate_vectors_by_eigenvector(w0)
    fn = mkdirs(os.path.join('%s'%path,'w0_similarity_kinet'))
    plt.figure()
    plt.pcolormesh(w0_similarity_matrix,vmin=-1,vmax=1,cmap='YlGnBu')
    plt.colorbar()
    plt.xlabel('index',fontsize=18)
    plt.xticks(fontsize=18)
    plt.ylabel('index',fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(fn,'%s'%nota))
    plt.close()

    w1_similarity_matrix = seperate_vectors_by_eigenvector(w1)
    fn = mkdirs(os.path.join('%s'%path,'w1_similarity_kinet'))
    plt.figure()
    plt.pcolormesh(w1_similarity_matrix,vmin=-1,vmax=1,cmap='YlGnBu')
    plt.colorbar()
    plt.xlabel('index',fontsize=18)
    plt.xticks(fontsize=18)
    plt.ylabel('index',fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(fn,'%s'%nota))
    plt.close()

    w2_similarity_matrix = seperate_vectors_by_eigenvector(w2)
    fn = mkdirs(os.path.join('%s'%path,'w2_similarity_kinet'))
    plt.figure()
    plt.pcolormesh(w2_similarity_matrix,vmin=-1,vmax=1,cmap='YlGnBu')
    plt.colorbar()
    plt.xlabel('index',fontsize=18)
    plt.xticks(fontsize=18)
    plt.ylabel('index',fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(fn,'%s'%nota))
    plt.close()

    w3_similarity_matrix = seperate_vectors_by_eigenvector(w3)
    fn = mkdirs(os.path.join('%s'%path,'w3_similarity_kinet'))
    plt.figure()
    plt.pcolormesh(w3_similarity_matrix,vmin=-1,vmax=1,cmap='YlGnBu')
    plt.colorbar()
    plt.xlabel('index',fontsize=18)
    plt.xticks(fontsize=18)
    plt.ylabel('index',fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(fn,'%s'%nota))
    plt.close()

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

def KINET_DSMC(x, v, a, rand_vec, training=False):
    dt = 1
    coll_coef = 0.9

    bs, chnl_old, h, w = x.shape
    _, chnl, _ = rand_vec.shape
    device = x.device 
    n = h * w # number of particles
    n_divide = chnl / chnl_old
    n = int(n / n_divide)
    a = a.reshape(bs, chnl, n)
    v = v.reshape(bs, chnl, n)
    x = x.reshape(bs, chnl, n)

    # v = v + a * dt
    v = a * dt

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

    if not training:
        coll_coef = 0

    collision_mask = mask > (1 -  coll_coef) # (bs, n, n), mask of particles that collide, equivalent to bernoulli(p=v_r/v_max*u_x)
    # print(collision_mask)
        
    delta_v = torch.zeros((bs, chnl, n, n))

    v_r_leaving = project_to_antisymmetric(v_r, chnl, rand_vec)

    delta_v = v_cm + v_r_leaving - v.unsqueeze(-1)

    v = v + torch.sum(delta_v * collision_mask.unsqueeze(1), dim=2)

    x = x + v * dt

    return x.view(bs, int(chnl/n_divide), h, w), v.view(bs, int(chnl/n_divide), h, w)

def KINET_DSMC2(x, v, a, rand_vec, training=False):
    dt = 1
    coll_coef = 0.9

    bs, chnl_old, h, w = x.shape
    _, chnl, _ = rand_vec.shape
    device = x.device 
    n = h * w # number of particles
    n_divide = chnl / chnl_old
    n = int(n / n_divide)
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

    return x.view(bs, int(chnl/n_divide), h, w), v.view(bs, int(chnl/n_divide), h, w)
################################################################
# fourier layer
################################################################

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        # compl_mul2d = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        # print('compl_mul2d', compl_mul2d.shape)

        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(12, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, v=None, rand_vec=None):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2) # (bs, width, d, d)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x, rand_vec

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    
class FNO2d_dropout(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d_dropout, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(12, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, v=None, rand_vec=None):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2) # (bs, width, d, d)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.dropout(x)
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.dropout(x)
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.dropout(x)
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = self.dropout(x)

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x, rand_vec

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


class FNO2d_kinet(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d_kinet, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(12, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, v=None, rand_vec=None):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2) # (bs, width, d, d)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        a = self.conv0(x) # acceleration
        p = self.w0(x) # position
        vec = rand_vec()
        x, v = KINET_DSMC(p, v, a, vec, self.training)
        x = F.gelu(x)
        v = F.gelu(v)

        a = self.conv1(x)
        p = self.w1(x)
        vec = rand_vec()
        x, v = KINET_DSMC(p, v, a, vec, self.training)
        x = F.gelu(x)
        v = F.gelu(v)

        a = self.conv2(x)
        p = self.w2(x)
        vec = rand_vec()
        x, v = KINET_DSMC(p, v, a, vec, self.training)
        x = F.gelu(x)
        v = F.gelu(v)

        a = self.conv3(x)
        p = self.w3(x)
        vec = rand_vec()
        x, _ = KINET_DSMC(p, v, a, vec, self.training)

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x, rand_vec

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class FNO2d_kinet2(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d_kinet2, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(12, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, v=None, rand_vec=None):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2) # (bs, width, d, d)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        a = self.conv0(x) # acceleration
        p = self.w0(x) # position
        vec = rand_vec()
        x, v = KINET_DSMC2(p, v, a, vec, self.training)
        x = F.gelu(x)
        v = F.gelu(v)

        a = self.conv1(x)
        p = self.w1(x)
        vec = rand_vec()
        x, v = KINET_DSMC2(p, v, a, vec, self.training)
        x = F.gelu(x)
        v = F.gelu(v)

        a = self.conv2(x)
        p = self.w2(x)
        vec = rand_vec()
        x, v = KINET_DSMC2(p, v, a, vec, self.training)
        x = F.gelu(x)
        v = F.gelu(v)

        a = self.conv3(x)
        p = self.w3(x)
        vec = rand_vec()
        x, _ = KINET_DSMC2(p, v, a, vec, self.training)

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x, rand_vec

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class FNO2d_VAEKINET(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d_VAEKINET, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(12, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0_mean = nn.Conv2d(self.width, self.width, 1)
        self.w0_var = nn.Conv2d(self.width, self.width, 1)
        self.w1_mean = nn.Conv2d(self.width, self.width, 1)
        self.w1_var = nn.Conv2d(self.width, self.width, 1)
        self.w2_mean = nn.Conv2d(self.width, self.width, 1)
        self.w2_var = nn.Conv2d(self.width, self.width, 1)
        self.w3_mean = nn.Conv2d(self.width, self.width, 1)
        self.w3_var = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, v=None, rand_vec=None):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2) # (bs, width, d, d)
        bs, chnl, h, w = x.shape
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        mean = self.w0_mean(x)
        var = self.w0_var(x)
        std = torch.exp(var / 2)
        eps = torch.randn(bs, chnl, h*w).to(x.device)
        eps = eps/ torch.norm(eps, dim=1, keepdim=True)
        eps = eps.view(bs, chnl, h, w)
        x2 = mean + eps * std
        x = x1 + x2
        x = F.tanh(x)

        x1 = self.conv1(x)
        mean = self.w1_mean(x)
        var = self.w1_var(x)
        std = torch.exp(var / 2)
        eps = torch.randn(bs, chnl, h*w).to(x.device)
        eps = eps/ torch.norm(eps, dim=1, keepdim=True)
        eps = eps.view(bs, chnl, h, w)
        x2 = mean + eps * std
        x = x1 + x2
        x = F.tanh(x)

        x1 = self.conv2(x)
        mean = self.w2_mean(x)
        var = self.w2_var(x)
        std = torch.exp(var / 2)
        eps = torch.randn(bs, chnl, h*w).to(x.device)
        eps = eps/ torch.norm(eps, dim=1, keepdim=True)
        eps = eps.view(bs, chnl, h, w)
        x2 = mean + eps * std
        x = x1 + x2
        x = F.tanh(x)

        x1 = self.conv3(x)
        mean = self.w3_mean(x)
        var = self.w3_var(x)
        std = torch.exp(var / 2)
        eps = torch.randn(bs, chnl, h*w).to(x.device)
        eps = eps/ torch.norm(eps, dim=1, keepdim=True)
        eps = eps.view(bs, chnl, h, w)
        x2 = mean + eps * std
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return x, rand_vec

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

################################################################
# configs
################################################################
parser = argparse.ArgumentParser(description='PyTorch FNO 2D')
parser.add_argument('--ntrain', type=int, default=20, help='number of training samples')
parser.add_argument('--ntest', type=int, default=20, help='number of testing samples')
parser.add_argument('--modes', type=int, default=12, help='number of Fourier modes')
parser.add_argument('--width', type=int, default=20, help='width of the neural network')
parser.add_argument('--batch_size', type=int, default=20, help='training batch size')
parser.add_argument('--batch_size2', type=int, default=20, help='testing batch size')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--scheduler_step', type=int, default=100, help='step size for learning rate scheduler')
parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='gamma for learning rate scheduler')
parser.add_argument('--model_name', type=str, default='FNO2d_dropout', help='model name: FNO2d, FNO2d_dropout, FNO2d_kinet, FNO2d_kinet2 or FNO2d_VAEKINET')
parser.add_argument('--s1', type=int, default=32, help='resolution of training data')
parser.add_argument('--s2', type=int, default=64, help='resolution of testing data')
parser.add_argument('--T_in', type=int, default=10, help='number of input time steps')
parser.add_argument('--T', type=int, default=10, help='number of output time steps')
args = parser.parse_args()

TRAIN_PATH = 'data/newheat32_data_1.mat'
TEST_PATH = 'data/newheat64_data_2.mat'
savepath = 'log/new-kinet-fno-heat-32-64/'+args.model_name

if not os.path.exists(savepath):
    os.makedirs(savepath)
    print('create folder:', savepath)
n_divide1 = 8
n_divide2 = 32
device = f'cuda:0'
if_wandb = False
if_plot = False
if_save = False

if if_wandb:
    # wandb.init(project="kinet-fno-ns-32-64", config=args, name=f"{args.model_name}")
    wandb.init(project="new-kinet-fno-heat-32-64", config=args, name=f"{args.model_name}")

print("epochs, learning_rate, scheduler_step, scheduler_gamma: ", args.epochs, args.lr, args.scheduler_step, args.scheduler_gamma)


# path = 'ns_fourier_2d_rnn_V10000_T20_N'+str(args.ntrain)+'_ep' + str(args.epochs) + '_m' + str(args.modes) + '_w' + str(args.width)
# path_model = 'model/'+path
# path_train_err = 'results/'+path+'train.txt'
# path_test_err = 'results/'+path+'test.txt'
# path_image = 'image/'+path

sub = 1
T_in = 10
T = 10
step = 1

################################################################
# load data
################################################################

reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('u')[:args.ntrain,::sub,::sub,:T_in]
train_u = reader.read_field('u')[:args.ntrain,::sub,::sub,T_in:T+T_in]

reader = MatReader(TEST_PATH)
test_a = reader.read_field('u')[-args.ntest:,::sub,::sub,:T_in]
test_u = reader.read_field('u')[-args.ntest:,::sub,::sub,T_in:T+T_in]

print('train_u.shape: ', train_u.shape)
print('test_u.shape: ', test_u.shape)
assert (args.s1 == train_u.shape[-2])
assert (T == train_u.shape[-1])

train_a = train_a.reshape(args.ntrain,args.s1,args.s1,T_in)
test_a = test_a.reshape(args.ntest,args.s2,args.s2,T_in)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=args.batch_size, shuffle=False)


################################################################
# training and evaluation
################################################################
if args.model_name == 'FNO2d':
    model = FNO2d(args.modes, args.modes, args.width)
    vec_train = None
    vec_test = None
    v1 = None
    v2 = None
if args.model_name == 'FNO2d_dropout':
    model = FNO2d_dropout(args.modes, args.modes, args.width)
    vec_train = None
    vec_test = None
    v1 = None
    v2 = None
elif args.model_name == 'FNO2d_kinet':
    model = FNO2d_kinet(args.modes, args.modes, args.width)
    vec_train = rand_vec(args.ntrain, args.width * n_divide1, args.s1 * args.s1 / n_divide1, device)
    vec_test = rand_vec(args.ntest, args.width * n_divide2, args.s2 * args.s2 / n_divide2, device)  
    v1 = torch.randn(args.ntrain, args.width, args.s1, args.s1).to(device)
    v2 = torch.randn(args.ntest, args.width, args.s2, args.s2).to(device)
elif args.model_name == 'FNO2d_kinet2':
    model = FNO2d_kinet2(args.modes, args.modes, args.width)
    vec_train = rand_vec(args.ntrain, args.width * n_divide1, args.s1 * args.s1 / n_divide1, device)
    vec_test = rand_vec(args.ntest, args.width * n_divide2, args.s2 * args.s2 / n_divide2, device)  
    v1 = torch.randn(args.ntrain, args.width, args.s1, args.s1).to(device)
    v2 = torch.randn(args.ntest, args.width, args.s2, args.s2).to(device)
elif args.model_name == 'FNO2d_VAEKINET':
    model = FNO2d_VAEKINET(args.modes, args.modes, args.width)
    vec_train = None
    vec_test = None
    v1 = None
    v2 = None
model.to(device)
# model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')

print('model params: ', count_params(model))
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

myloss = LpLoss(size_average=False)
for ep in range(args.epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            # print(xx.shape)
            y = yy[..., t:t + step]
            im, vec_train = model(xx, v1, vec_train)
            loss += myloss(im.reshape(args.batch_size, -1), y.reshape(args.batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)

        # print(' ')
        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(args.batch_size, -1), yy.reshape(args.batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im, vec_test = model(xx, v2, vec_test)
                loss += myloss(im.reshape(args.batch_size, -1), y.reshape(args.batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(args.batch_size, -1), yy.reshape(args.batch_size, -1)).item()

    t2 = default_timer()
    scheduler.step()
    print('epoch:', ep, 'time: %.2f' % (t2 - t1), 'train l2 step: %.8f' % (train_l2_step / args.ntrain / (T / step)), 'train l2 full: %.8f' % (train_l2_full / args.ntrain), 'test l2 step: %.8f' % (test_l2_step / args.ntest / (T / step)), 'test l2 full: %.8f' % (test_l2_full / args.ntest))
    if if_wandb:
        wandb.log({"train_l2_step": train_l2_step / args.ntrain / (T / step), "train_l2_full": train_l2_full / args.ntrain, "test_l2_step": test_l2_step / args.ntest / (T / step), "test_l2_full": test_l2_full / args.ntest})

    if if_plot:
        para_now = copy.deepcopy(model.state_dict())
        plot_weight_heatmap_eigen(para_now, savepath, nota='%s'%ep)

    if if_save:
        ckpt_path = os.path.join(savepath, f"checkpoint/checkpoint_epoch_{ep}.pth")
        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),     # 保存权重和偏置 :contentReference[oaicite:0]{index=0}
        }, ckpt_path)
    # print(ep, t2 - t1, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),
    #       test_l2_full / ntest)
# torch.save(model, path_model)

# pred = torch.zeros(test_u.shape)
# index = 0
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
# with torch.no_grad():
#     for x, y in test_loader:
#         test_l2 = 0;
#         x, y = x.cuda(), y.cuda()
#
#         out = model(x)
#         out = y_normalizer.decode(out)
#         pred[index] = out
#
#         test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
#         print(index, test_l2)
#         index = index + 1

# scipy.io.savemat('pred/'+path+'.mat', mdict={'pred': pred.cpu().numpy()})