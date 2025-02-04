import torch
import numpy as np
import pandas as pd
import re
from pathlib import Path
from pymatgen.core.structure import Element
from torch_geometric.data import Batch, Data
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from dklbo.model.dkl import DKLCompGraph
from dklbo.data.util import standalize, ElementDescTable, normalize



import warnings
warnings.filterwarnings("ignore")
""" parameters """
n_ini = 10 # number of initial samples
n_gp = 10 # number of
beta = 0.2 # beta for UCB


""" data load"""
path_to_data = Path('../data/datasets/experiment/exp_bandgap.xlsx')
df = pd.read_excel(path_to_data, index_col=0, header=0)
columns = df.columns.tolist()[0:-1]
x_table = df.iloc[:, 0:-1].values
y_table = df.iloc[:, -1].values
A_sites = ['FA', 'MA', 'Cs']
B_sites = ['Pb', 'Sn']
C_sites = ['Br', 'Cl', 'I']
follow_batch = ['x_asite', 'x_bsite', 'x_csite']
def get_elem_desc(elem):
    def get_desc(elem):
        e = Element(elem)
        e_aff = e.electron_affinity  # electron affinity
        e_X = e.X  # electron negativity
        r_atom = e.data['Atomic radius']
        valence = e.electronic_structure.split('.')
        s_ele = 0
        p_ele = 0
        for v in valence:
            if bool(re.match('.[a-z][0-9]', v)):
                v = v[1:]
                v = list(v)
                if v[0] == 's':
                    s_ele = int(v[1])
                elif v[0] == 'p':
                    p_ele = int(v[1])
                else:
                    pass
            else:
                pass

        p = [e_aff,
             e_X,
             r_atom,
             s_ele,
             p_ele,
             ]
        return p

    if elem == 'MA':  # methylammonium  (CH3NH3+)
        p = [get_desc(e) for e in ['C', 'H', 'H', 'H', 'N', 'H', 'H', 'H']]
        p = np.array(p)
        p = np.mean(p, axis=0)
    elif elem == 'FA':  # formamidinium (NH2CHNH2+)
        p = [get_desc(e) for e in ['N', 'H', 'H', 'C', 'H', 'N', 'H', 'H']]
        p = np.array(p)
        p = np.mean(p, axis=0)
    else:
        p = np.array(get_desc(elem))

    return p

edt = ElementDescTable(
    composition_columns = columns,
    ABC=[A_sites, B_sites, C_sites],
    n_comb=[3, 2, 3],
    desc_func=get_elem_desc)
print(edt.prop_dict)
graphs = edt.collate_graphs_from_tablevalues(x_table, y_table,
                                             follow_batch=follow_batch)

print(graphs)

""" neural network setting"""
n_abc_site = 3
n_conv = 2
n_feas = [graphs.x_asite.shape[1],graphs.x_bsite.shape[1],graphs.x_csite.shape[1]]
layers_conv = []
layers_pooling = []
for _ in range(n_abc_site):
    l = [n_feas[_]] * 3
    layers_conv.append([l for i in range(n_conv)])

    l = [n_feas[_],1]
    layers_pooling.append(l)
nn_param = {}
nn_param['n_abc_site'] = n_abc_site
nn_param['n_conv'] = n_conv
nn_param['layers_atom'] = []
nn_param['layers_conv'] = layers_conv
nn_param['layers_pooling'] = layers_pooling
nn_param['layers_fc'] = [sum(n_feas),n_gp]
nn_param['activation'] = torch.nn.Mish()
nn_param['follow_batch'] = follow_batch

lr = 0.001
epochs = 50


""" initial datasets"""
np.random.seed(0)
indx = np.arange(graphs.num_graphs)
target_indx = np.argmax(graphs.y.numpy())
indx = indx[indx!=target_indx]
np.random.shuffle(indx)
obs_indx = torch.tensor(indx[0:n_ini].tolist(),dtype=torch.long)
cand_indx = torch.tensor(np.append(indx[n_ini:],target_indx),dtype=torch.long)
target_indx = torch.tensor(target_indx, dtype=torch.long)

convergence = True
cycle = 0
while convergence:
    # make obserbed(train) and candidate(example) batch
    obs_batch = Batch.from_data_list(graphs.index_select(obs_indx),follow_batch=follow_batch)
    cand_batch = Batch.from_data_list(graphs.index_select(cand_indx),follow_batch=follow_batch)

    # standalize observed y
    y_standalized, stan_val = standalize(obs_batch.y)
    obs_batch.y = y_standalized.view(-1)

    # construct model
    likelihood = GaussianLikelihood()
    model = DKLCompGraph(train_x=None, # always None
                         train_y=obs_batch.y,
                         likelihood=likelihood,
                         nn_param=nn_param,
                         batch=obs_batch,
                         follow_batch=follow_batch,
                         noise_fix=False, # adjust noise
                         )

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
