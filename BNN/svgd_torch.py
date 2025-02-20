import numpy as np
import torch
device = 'cuda'

def SVGD_torch(X_list, V, dV, beta, T, lambdak, stepsize,bandwidth):
    grad_V = dV(X_list)+lambdak*torch.sign(X_list)
    nx, d = X_list.shape
    device = X_list.device 

    xk_i = X_list.unsqueeze(1)  # Shape: [nx, 1, d]
    xk_j = X_list.unsqueeze(0)  # Shape: [1, nx, d]
    # Compute rhoxtmp
    diff_jy = (xk_i - xk_j)  # Shape: [nx, nx, d]
    L2diff_jy = (diff_jy**2).sum(dim=2) 
    expdiff_jy = torch.exp(-L2diff_jy /(2*bandwidth))  # Shape: [nx, nx]
    kernel_SVGD = torch.zeros(nx,d,device = device)
    # Compute kernel SVGD
    for jx in range(nx):
    # gradient with respet to the first variable
        term1 = -expdiff_jy[:, jx].unsqueeze(1) * grad_V  # Shape: [nx, d]
        term2 = -diff_jy[:, jx, :] * (expdiff_jy[:, jx] / bandwidth).unsqueeze(1)  # Shape: [nx, d]
        kernel_SVGD[jx, :] = (term1 + term2).sum(dim=0)  # Shape: [d]

    X_list = X_list + stepsize/nx*kernel_SVGD
    X_list.detach_()
    return X_list


def ULA_torch(X_list, V, dV, beta, T, lambdak, stepsize):
    grad_V = dV(X_list)
    std = np.sqrt(2*stepsize)
    device = X_list.device 
    X_list = torch.sign(X_list) * torch.maximum(torch.abs(X_list) - lambdak * stepsize, torch.tensor(0.0, device=device))- stepsize * grad_V + std * torch.randn(X_list.shape, device= device)
    X_list.detach_()
    return X_list
