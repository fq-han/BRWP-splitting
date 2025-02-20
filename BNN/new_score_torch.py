import numpy as np
import torch
device = 'cuda'

def L1_score_torch(xk, lambdak, beta, T, bandwidth, ifproduct):
    device = xk.device
    nx, d = xk.shape

    # Compute Sk
    Sk = torch.sign(xk) * torch.maximum(torch.abs(xk) - lambdak * T, torch.tensor(0.0, device=device))
    
    xk_i = xk.unsqueeze(1)  # Shape: [nx, 1, d]
    xk_j = xk.unsqueeze(0)  # Shape: [1, nx, d]
    # Compute Nxi
    Nxi = torch.exp(beta / 2 * (lambdak * torch.abs(Sk) + (Sk - xk) ** 2 / (2 * T)))

    # # Compute rhoxtmp
    diff_jy = xk_i - xk_j  # Shape: [nx, nx, d]
    rhoxtmp = torch.exp(-beta / 2 * (diff_jy ** 2 / (2 * T))) * Nxi #shape: [nx,nx,d]
    # if torch.all(torch.isnan(rhoxtmp)):
    #     raise Exception
     
    # # Compute rhoxj and drhoxj
    drhoxjl = torch.zeros(nx,d, device=device)
    
    rhoxjl = rhoxtmp.prod(dim = -1)# Shape: [nx, nx]
    for jx in range(nx):
          rhoxjld = rhoxjl[:,jx]
          drhoxjl[jx,:] = (xk*rhoxjld.unsqueeze(-1)).sum(dim = 0)
    rhoxjl = rhoxjl.sum(dim=1,keepdim=True)  # Shape: [nx, d] last dimension constant

    score = drhoxjl / rhoxjl  # Broadcasting: [nx, d]
    return score


def one_step_splitting_torch(X_list, V, dV, beta, T, lambdak, stepsize):
    """
    Args:
        X_list (torch.Tensor): List of current points to evolve. Shape [N, d]
        V (function): Potential function
        dV (function): Derivative of potential function
        beta (float): Regularization parameter
        T (float): Time to evolve towards
        lambdak (float): Lambda regularization parameter
        stepsize (float): Step size
    """
    score = L1_score_torch(X_list, lambdak, beta, stepsize, bandwidth = 0.5, ifproduct = 1)
    
    grad_V = dV(X_list)
    # Update X_list using the defined rules
    X_list = X_list - stepsize * grad_V
    X_list = X_list + 1/2 * torch.sign(X_list) * torch.maximum(torch.abs(X_list) - lambdak * stepsize, torch.tensor(0.0, device=device)) - 1/2 * score
    X_list.detach_()
    return X_list