import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.data import Batch, Data
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from dklbo.data.util import CIFData
from dklbo.model.dkl import DKLCGCNN
from dklbo.data.util import standalize

import warnings
warnings.filterwarnings("ignore")

""" parameters """
n_ini = 10 # number of initial samples
n_gp = 15 # number of
beta = 0.2 # beta for UCB


""" data load"""
path_to_data = Path('../data/datasets/calculation')
print(path_to_data)
dataset = CIFData(path_to_data, 1)
graphs = []
for x, y, id in dataset:
    edge_index = []
    for n in range(len(x[2])):
        e = x[2][n].tolist()
        N = len(e)
        n = [n] * N
        edge_index.append(torch.tensor([n,e],dtype=torch.long))

    shape = x[1].shape
    edge_attr = np.reshape(x[1],(shape[0]*shape[1],shape[2]))
    edge_attr = torch.tensor(edge_attr.tolist(),dtype=torch.float32)
    edge_index = torch.cat(edge_index,dim=1)
    x = torch.tensor(x[0].tolist(), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,y = y))
graphs = Batch.from_data_list(graphs)

""" neural network setting"""
nn_param = {}
nn_param['orig_atom_fea_len'] = graphs.x.shape[-1]
nn_param['nbr_fea_len'] = graphs.edge_attr.shape[-1]
nn_param['atom_fea_len'] = 32
nn_param['n_conv'] = 3
nn_param['n_h'] = 1
nn_param['h_fea_len'] = n_gp # input dimension for GP part
lr = 0.01
epochs = 50


""" initial datasets"""
np.random.seed(0)
indx = np.arange(graphs.num_graphs)
target_indx = np.argmax(graphs.y.numpy())
indx = indx[indx!=target_indx]
np.random.shuffle(indx)
obs_indx = torch.tensor(indx[0:n_ini].tolist(),dtype=torch.long)
cand_indx = torch.tensor(np.append(indx[n_ini:],target_indx),dtype=torch.long)

convergence = True
cycle = 0
while convergence:
    # make obserbed(train) and candidate(example) batch
    obs_batch = Batch.from_data_list(graphs.index_select(obs_indx))
    cand_batch = Batch.from_data_list(graphs.index_select(cand_indx))

    # standalize observed y
    y_standalized, stan_val = standalize(obs_batch.y)
    obs_batch.y = y_standalized.view(-1)

    # construct model
    likelihood = GaussianLikelihood()
    model = DKLCGCNN(None, obs_batch.y, likelihood, nn_param,
                     obs_batch, noise_fix=True)

    # training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)  # compute marginal (log) likehood
    model.train()
    model.likelihood.train()
    model.trainmodel(optimizer,scheduler,mll, epochs)

    # select a next candidate
    model.eval()
    model.likelihood.eval()
    pred = model.likelihood(model(cand_batch, follow_batch=follow_batch))
    mean = pred.mean
    std = pred.stddev
    ucb = mean + beta * std
    next_cand_indx = cand_indx[torch.argmax(ucb)]

    print('----------cycle %s-----------' %(cycle))
    print((next_cand_indx,next_cand_indx.shape),(target_indx,target_indx.shape))
    cycle += 1
    print(len(obs_indx),len(cand_indx))
    if next_cand_indx == target_indx:
        convergence = False
    else:
        # update obserbed indices
        obs_indx = torch.cat([obs_indx, next_cand_indx.unsqueeze(0)])
        cand_indx = cand_indx[cand_indx != next_cand_indx]
print('converged cycle',cycle)
