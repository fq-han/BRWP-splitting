#This code is based Hongye Tan's BNN implementaion
import numpy as np
import torch
import torch.nn as nn
import argparse
import bnn_utils
import os
import wass_opt_lib_torch
import svgd_torch
import new_score_torch
import torch.autograd as autograd
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt

# loggers
import logging
device = 'cuda'
base_checkpoint_path = 'uci/checkpoints'
os.makedirs(base_checkpoint_path, exist_ok = True)

dataset_paths = {
    "boston": 'uci/raw_data/housing/data',
    "power": 'uci/raw_data/power/data',
    "concrete": 'uci/raw_data/concrete/data',
    "wine": 'uci/raw_data/wine/data',
    "kin8nm": 'uci/raw_data/kin8nm/data',
    "energy": 'uci/raw_data/energy/data',
    "protein": 'uci/raw_data/protein/data',
    "naval": 'uci/raw_data/naval/data'
}

train_params = {
    "boston":   {'n_epochs':500, 'batch_size': 100, 'stepsize': 1*1e-2,'T': 1*1e-2},
    "power": {'n_epochs': 500, 'batch_size': 100, 'stepsize': 1e-2,'T': 1e-2}, 
    "concrete": {'n_epochs': 500, 'batch_size': 100, 'stepsize': 1*1e-2,'T': 1*1e-2}, 
    "wine":     {'n_epochs': 500, 'batch_size': 100, 'stepsize': 1e-2,'T': 1e-2},
    "kin8nm":   {'n_epochs': 500, 'batch_size': 100, 'stepsize': 1e-2,'T': 1e-2}, 
    "energy":   {'n_epochs': 500, 'batch_size': 100, 'stepsize': 1*1e-2,'T': 1*1e-2}, 
    "protein":   {'n_epochs': 500 , 'batch_size': 100, 'stepsize': 1e-2,'T': 1e-2}, 
    "naval":   {'n_epochs': 500, 'batch_size': 100, 'stepsize': 1*1e-2,'T': 1*1e-2} 
}


def main():
    splitting_stepsize, brwp_stepsize, ula_stepsize, svgd_stepsize =  5*1e-3, 5*1e-3, 5*1e-3, 0.25
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='energy', help="dataset type, boston|power|concrete|wine|kin8nm|energy|protein|naval")
    parser.add_argument('--nn_init_num', type=int, default=100, help="Number of particles")
    parser.add_argument('--trial', type=int, default=0, help="trial number to average over")
    parser.add_argument('--T', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=1.)
    parser.add_argument('--stepsize', type=float, default=-1.)
    parser.add_argument('--loggingprefix', type=str, default="gridsearch")
    args = parser.parse_args()

    key = args.data
    if args.stepsize > 0:
        train_params[key]['stepsize'] = args.stepsize

    logging.basicConfig(filename='logs/bnn_'+args.loggingprefix+'_'+key+'.log', filemode='a', 
                        format='%(name)s - %(levelname)s - %(message)s', level=logging.WARNING)
    logger = logging.getLogger('bnn')
    logger.setLevel(logging.INFO)
    logger.info("experiment {}({}), stepsize {}, T {}, beta {}".format(
        key, args.trial, train_params[key]['stepsize'], args.T, args.beta))

    print("key", key)
    train_dl, test_dl, num_features, targ_std, targ_mean, data_1k, targ_1k, n_train = bnn_utils.load_dataset(
        dataset_paths[key], train_params[key]['batch_size'])
 
    lambdak = 0.1/n_train

    data_1k = data_1k.to(device)
    targ_1k = targ_1k.to(device)
    tbar = tqdm(train_dl, ncols=80, position=0, leave=True)
    tbar2 = tqdm(range(1, train_params[key]['n_epochs']+1), ncols=120, position=1)
    net_list = bnn_utils.initialize_networks(num_features, device=device, nn_init_num=args.nn_init_num)

    print("initialized model")
    net_params = torch.vstack([bnn_utils.model_to_params(model) for model in net_list])
    net_params_splitting = net_params
    net_params_brwp = net_params
    net_params_ula = net_params
    net_params_svgd = net_params
    print("begin train")

    # Storage for MMSE and RMSE values for three methods
    metrics = {
        'epoch': [],
        'mmse_splitting': [], 'rmse_splitting': [],  'logl_splitting': [], 'neg_var_splitting': [],
        'mmse_brwp': [], 'rmse_brwp': [],  'logl_brwp': [], 'neg_var_brwp': [],
        'mmse_ula': [], 'rmse_ula': [], 'logl_ula': [], 'neg_var_ula': [],
        'mmse_svgd': [], 'rmse_svgd': [], 'logl_svgd': [], 'neg_var_svgd': [],
    }

    for epoch in tbar2:
        for batch_idx, (dat_, targ_) in enumerate(tbar):
            dat = dat_.to(device)
            targ = targ_.to(device)

            V = partial(bnn_utils.nllhood, features=dat, y_true=targ, n_features=num_features, n_train=n_train,lambdak= lambdak)

            def dV(params):
                params.requires_grad_(True)
                return autograd.grad(V(params), params, grad_outputs=torch.ones(args.nn_init_num).to(device))[0]

            # # First update method
            net_params_splitting = new_score_torch.one_step_splitting_torch(
                net_params_splitting, V, dV, args.beta, splitting_stepsize, lambdak = lambdak, stepsize=splitting_stepsize
            )

            # # Second update method
            net_params_brwp = wass_opt_lib_torch.update_once(
                net_params_brwp, V, dV, args.beta, brwp_stepsize, lambdak = lambdak,stepsize=brwp_stepsize, sample_iters=50
            )

            # # # Third update method
            net_params_ula = svgd_torch.ULA_torch(
                net_params_ula, V, dV, args.beta, ula_stepsize, lambdak = lambdak, stepsize=ula_stepsize
            )

            # Fourth update method
            net_params_svgd = svgd_torch.SVGD_torch(
                net_params_svgd, V, dV, args.beta, svgd_stepsize, lambdak = lambdak, stepsize=svgd_stepsize, bandwidth = 1
            )

        if epoch % 5 == 0:
            # Validation for each method
            for method, params in zip(
                ['splitting', 'brwp', 'ula','svgd'], 
                [net_params_splitting, net_params_brwp, net_params_ula,net_params_svgd]
            ):
                mmse_total, ll_total, var_total = 0, 0, 0
                for dat_, targ_ in test_dl:
                    dat = dat_.to(device)
                    targ = targ_.to(device)
                    mmse, ll, neg_var = bnn_utils.mmse_logprior(
                        params, dat, targ, num_features, targ_std, targ_mean, data_1k, targ_1k, lambdak
                    )               
                    ll_total += ll.item()
                    mmse_total += mmse.item()
                    var_total += neg_var.item()
                mmse_full = mmse_total / len(test_dl)
                rmse_full = np.sqrt(mmse_full)
                logl_full = ll_total
                var_full = var_total/len(test_dl)

                # Store metrics
                metrics[f'mmse_{method}'].append(mmse_full)
                metrics[f'rmse_{method}'].append(rmse_full)
                metrics[f'logl_{method}'].append(logl_full)
                metrics[f'neg_var_{method}'].append(var_full)

            tbar2.set_description(
                 'epoch {}, logl_splitting {:.4f}, rmse_splitting {:.4f}, logl_brwp {:.4f}, rmse_brwp {:.4f},\n'.format(
                 epoch, metrics['logl_splitting'][-1], metrics['rmse_splitting'][-1], 
                 metrics['logl_brwp'][-1], metrics['rmse_brwp'][-1]))
            tbar2.set_description(
                 'logl_ula {:.4f}, rmse_ula {:.4f}, logl_svgd {:.4f}, rmse_svgd {:.4f}'.format(
                 metrics['logl_ula'][-1], metrics['rmse_ula'][-1], 
                 metrics['logl_svgd'][-1], metrics['rmse_svgd'][-1]))
        tbar.reset()
    
    # print('Vx',V(net_params_splitting))
    # print('L1',lambdak*torch.abs(net_params_splitting).sum(axis = 1))
    # print('Vx',V(net_params_svgd))
    # print('L1',lambdak*torch.abs(net_params_svgd).sum(axis = 1))

    # Save metrics to a file
    metrics_path = os.path.join('metrics', key + '_metrics.npy')
    os.makedirs('metrics', exist_ok=True)
    np.save(metrics_path, metrics)

    epochs_sequence = list(range(5, train_params[key]['n_epochs'] + 1, 5))
    # Plot Log likelihood
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_sequence, metrics['logl_splitting'], label='logl (Splitting)', color='blue')
    plt.plot(epochs_sequence, metrics['logl_brwp'], label='logl (BRWP)', color='green')
    plt.plot(epochs_sequence, metrics['logl_ula'], label='logl (ULA)', color='orange')
    plt.plot(epochs_sequence, metrics['logl_svgd'], label='logl (SVGD)', color='black')

    plt.xlabel('Epoch')
    plt.ylabel('logl')
    plt.legend()
    plt.title('logl over Epochs')
    plt.grid()
    plt.savefig(os.path.join('metrics', key + '_logl_plot.png'))
    plt.show()

    # Plot RMSE
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_sequence, metrics['rmse_splitting'], label='RMSE (Splitting)', linestyle='dashed', color='blue')
    plt.plot(epochs_sequence, metrics['rmse_brwp'],label='RMSE (BRWP)', linestyle='dashed', color='green')
    plt.plot(epochs_sequence, metrics['rmse_ula'],  label='RMSE (ULA)', linestyle='dashed', color='orange')
    plt.plot(epochs_sequence, metrics['rmse_svgd'],  label='RMSE (SVGD)', linestyle='dashed', color='black')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE') 
    plt.legend()
    plt.title('RMSE over Epochs')
    plt.grid()
    plt.savefig(os.path.join('metrics', key + '_rmse_plot.png'))
    plt.show()


    logger.info("logl (Splitting) {}, RMSE (Splitting) {}, var (Splitting) {}".format(metrics['logl_splitting'][-1], metrics['rmse_splitting'][-1], metrics['neg_var_splitting'][-1]))
    logger.info("logl (BRWP) {}, RMSE (BRWP) {}, var (BRWP) {}".format(metrics['logl_brwp'][-1], metrics['rmse_brwp'][-1], metrics['neg_var_brwp'][-1]))
    logger.info("logl (ULA) {}, RMSE (ULA) {}, var (ULA) {}".format(metrics['logl_ula'][-1], metrics['rmse_ula'][-1], metrics['neg_var_ula'][-1]))
    logger.info("logl (SVGD) {}, RMSE (SVGD) {}, var (SVGD) {}".format(metrics['logl_svgd'][-1], metrics['rmse_svgd'][-1], metrics['neg_var_svgd'][-1]))
 


if __name__ == "__main__":
    main()