import numpy as np
import torch
# Update via 
#    dX/dt = -\nabla f(x) - \beta \nabla \log(\rho(x))
# where \rho is the score funciton

device = 'cuda'

def compute_normalizing_constant(y, V, beta, T,lambdak, n_iter = 25):
    # Numerically compute
    # \int \exp(-1/(2\beta) (V(z) + ||z-y||^2/(2T)) dz
    # = (4pi T beta)^(d/2) E_{z ~ N(y, 2 T beta I)}[exp(-1/(2beta) V(z))]
    # Takes the form of a Gaussian expectation
    # can accept vectorized
    # assume y is batched as in [N, d], then returns [N, 1]

    dims = y.shape[-1]
    variance = 2 * T * beta
    std = np.sqrt(variance)
    # front_constant = (2*np.pi*variance)**(dims/2)
    front_constant = 1
    avg = None
    for i in range(n_iter):
        # sample z from normal distribution z ~ N(y, 2 T beta I)
        z = y + std * torch.randn_like(y)
        current = torch.exp(-(V(z)+lambdak*torch.abs(z).sum(dim=-1))/(2*beta))
        if avg is None:
            avg = current
        else:
            avg = avg + current

    return avg/n_iter * front_constant



def compute_score(W_list, V, dV, beta, T, lambdak,sample_iters = 25, compute_energy = False, **kwargs):
    # rho_0 = \sum_i delta_{w_i}
    # W_list = [w_i : i = 0,...,N], shape = [N, d]
    # assume V can accept batched input V : [N, d] -> [N, 1]
    # dV : [N, d] -> [N, d]
    batch_num = W_list.shape[0]
    dims = W_list.shape[-1]
    mat_differences = - (W_list[None,:,:]).repeat(batch_num,1,1) \
                    + (W_list[:,None,:]).repeat(1,batch_num,1) # of shape [N,N,d]. mat_differences[i,j] = w_i - w_j
    # print("mat_diff", mat_differences.shape)

    squared_differences = torch.sum(mat_differences**2, dim=2) # of shape [N,N]
    
    # approximate normalizing constant for K using MC
    with torch.no_grad():
        normalizing_constants = compute_normalizing_constant(W_list, V, beta, T, lambdak,n_iter = sample_iters)
 
    # compute the V
    V_arr = V(W_list) + lambdak*torch.abs(W_list).sum(dim = -1) # [N, 1]
    # enforce [N, 1]. Required for broadcasting later.
    if len(V_arr.shape) == 1:
        V_arr = V_arr[:, None]
    elif len(V_arr.shape) > 2:
        raise Exception("V is not outputting a scalar?")
    dV_arr = dV(W_list) + torch.sign(W_list) # [N, d]

    # for computing the exponential
    unscaled_density = torch.exp(-1/(2*beta) * (V_arr + squared_differences/(2*T))) # [N,N]
    # print("us", unscaled_density.shape)
    scaled_density = unscaled_density /(normalizing_constants[None,:]) # [N,N]. normalizing constant.T is [1,N] for Z(w_j)\
    # matrix of scaled density K(w_i, w_j)
    
    # print("s", scaled_density.shape)
    rho = torch.sum(scaled_density, dim=1) # to get integral with rho_0, sum over j

    # for computing the score d\rho
    # compute pre-multiplier
    # print(dV_arr.shape)
    # print(mat_differences.shape)
    pre_multiplier = -1/(2*beta) * (dV_arr[:, None, :] + mat_differences/T) # should be of the shape [N,N,d]

    score_unsummed = pre_multiplier * scaled_density[:,:,None]
    score = torch.sum(score_unsummed, dim=1) # technically only \grad \rho(w_i), not the score.
    return rho, score # rho: [N]. score: [N,d]
    

def update_once(X_list, V, dV, beta, T, lambdak, stepsize, sample_iters=25, compute_energy=False):
    """_summary_

    Args:
        X_list (list of np.array): List of current points to evolve. Shape [N,d]
        V (function): Potential function
        dV (function): Derivative of potential function
        beta (float): Regularization parameter
        T (float): time to evolve towards
    """
    if len(X_list.shape) == 1: #assume it is 1d.
        X_list = X_list[:, None]
    elif len(X_list.shape) != 2:
        raise Exception("Wrong shape for input")

    if beta <= 0:
        raise Exception("Beta must be positive")
    if T <= 0:
        raise Exception("T must be positive")
    if compute_energy:
        rho, score, energy = compute_score(X_list, V, dV, beta, T,lambdak, sample_iters = sample_iters, compute_energy=compute_energy) #rho: [N]. score: [N,d]
    else:
        rho, score = compute_score(X_list, V, dV, beta, T,lambdak, sample_iters = sample_iters, compute_energy=compute_energy) #rho: [N]. score: [N,d]

    # print("rs shape", rho.shape, score.shape)
    grad_V = dV(X_list)+torch.abs(X_list)

    # last term is d_x log rho
    X_list = X_list - stepsize * grad_V - beta * stepsize * score/(rho[:, None])
    X_list.detach_()
    if compute_energy:
        return X_list, energy
    else:
        return X_list