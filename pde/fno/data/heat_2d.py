import torch

import math

# import matplotlib.pyplot as plt
# import matplotlib
# from drawnow import drawnow, figure

from my_random_fields import GaussianRF

from timeit import default_timer

import scipy.io

#T0: initial temperature field
#Q: forcing term
#alpha: thermal diffusivity coefficient
#T: final time
#delta_t: internal time-step for solve (descrease if blow-up)
#record_steps: number of in-time snapshots to record
def heat_conduction_2d(T0, Q, alpha, T, delta_t=1e-4, record_steps=1):

    #Grid size - must be power of 2
    N = T0.size()[-1]

    #Maximum frequency
    k_max = math.floor(N/2.0)

    #Number of steps to final time
    steps = math.ceil(T/delta_t)

    #Initial temperature field to Fourier space
    T_h_complex = torch.fft.fft2(T0, norm=None)
    T_h = torch.view_as_real(T_h_complex)
    # print('w_h', w_h.shape)

    #Forcing to Fourier space
    Q_h_complex = torch.fft.fft2(Q, norm=None)
    Q_h = torch.view_as_real(Q_h_complex)
    # print('f_h', f_h.shape)

    #If same forcing for the whole batch
    if len(Q_h.size()) < len(T_h.size()):
        Q_h = torch.unsqueeze(Q_h, 0)

    #Record solution every this number of steps
    record_time = math.floor(steps/record_steps)

    #Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=T0.device), torch.arange(start=-k_max, end=0, step=1, device=T0.device)), 0).repeat(N,1)
    #Wavenumbers in x-direction
    k_x = k_y.transpose(0,1)
    #Negative Laplacian in Fourier space
    lap = 4*(math.pi**2)*(k_x**2 + k_y**2)
    lap[0,0] = 1.0

    # Precompute spectral coefficients [1](@ref)
    cni_factor = (1 - 0.5 * alpha * delta_t * lap) / (1 + 0.5 * alpha * delta_t * lap)  # Crank-Nicolson integration factor
    source_factor = delta_t / (1 + 0.5 * alpha * delta_t * lap)  # Heat source scaling factor

    #Saving solution and time
    sol = torch.zeros(*T0.size(), record_steps, device=T0.device)
    sol_t = torch.zeros(record_steps, device=T0.device)

    #Record counter
    c = 0
    #Physical time
    t = 0.0
    for j in range(steps):
        # Update temperature field in Fourier space [1](@ref)
        T_h[..., 0] = cni_factor * T_h[..., 0] + source_factor * Q_h[..., 0]
        T_h[..., 1] = cni_factor * T_h[..., 1] + source_factor * Q_h[..., 1]

        #Update real time (used only for recording)
        t += delta_t

        if (j+1) % record_time == 0:
            #Solution in physical space
            T_complex = torch.view_as_complex(T_h)
            T = torch.fft.irfft2(T_complex, s=(N, N), norm=None)

            #Record solution and time
            sol[...,c] = T
            sol_t[c] = t

            c += 1


    return sol, sol_t


device = torch.device('cuda')

#Resolution
s = 32

#Number of solutions to generate
N = 20

#Set up 2d GRF with covariance parameters
GRF = GaussianRF(2, s, alpha=2.5, tau=7, seed=2, device=device)

#Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
t = torch.linspace(0, 1, s+1, device=device)
t = t[0:-1]

X,Y = torch.meshgrid(t, t)

Q = torch.zeros_like(X)
q = 0.1
Q[:, 0] = q  
Q[:, -1] = q  
Q[0, :] = q   
Q[-1, :] = q

#Number of snapshots from solution
record_steps = 200

#Inputs
a = torch.zeros(N, s, s)
#Solutions
u = torch.zeros(N, s, s, record_steps)

#Solve equations in batches (order of magnitude speed-up)

#Batch size
bsize = 20

alpha = 1e-4

c = 0
t0 = default_timer()
for j in range(N//bsize):

    #Sample random feilds
    T0 = GRF.sample(bsize)

    #Solve NS
    sol, sol_t = heat_conduction_2d(T0, Q, alpha, 50.0, 1e-4, record_steps)

    a[c:(c+bsize),...] = T0
    u[c:(c+bsize),...] = sol

    c += bsize
    t1 = default_timer()
    print(j, c, t1-t0)

# a.shape = [N, s, s], u.shape = [N, s, s, record_steps], t.shape = [record_steps]
scipy.io.savemat('data/newheat32_data_2.mat', mdict={'a': a.cpu().numpy(), 'u': u.cpu().numpy(), 't': sol_t.cpu().numpy()})