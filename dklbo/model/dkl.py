import torch
from abc import ABC, abstractmethod
from dklbo.model.modules import ExactGP_graph
from dklbo.model.crystal import CrystalGraphConvNet
from dklbo.model.compsition import CompGraphConvNet
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.priors.torch_priors import GammaPrior
import gpytorch
from torch_geometric.data import Batch, Data






class DKLCompGraph(ExactGP_graph):
    def __init__(self, train_x, train_y, likelihood, nn_param,
                 batch:Data, follow_batch, noise_fix=False):
        """
        :param train_x: dammy arg, should be None
        :param train_y: (N,) - Tensor
        :param nn_param: dict including keys listed below
            n_abc_site : int
            n_conv : int
            layers_conv: List of int
            layers_pooling: List of int
            layers_fc: List of int
            activation: activation function
        :param batch: Data(graph)
        :param follow_batch: list of keys for follow batch
        """


        super(DKLCompGraph, self).__init__(None, train_y.flatten(), likelihood)

        # transformer
        self.transformer = CompGraphConvNet(
            n_abc_site=nn_param['n_abc_site'],
            n_conv=nn_param['n_conv'],
            layers_atom=nn_param['layers_atom'],
            layers_fc=nn_param['layers_fc'],
            layers_pooling=nn_param['layers_pooling'],
            layers_conv=nn_param['layers_conv'],
            activation=nn_param['activation']
        )

        self.train_inputs = batch
        self.train_targets = train_y.flatten()

        n_output_kernel = nn_param['layers_fc'][-1]
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel=MaternKernel(
                nu=2.5,
                ard_num_dims=n_output_kernel,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
            outputscale_prior=GammaPrior(2.0, 0.15),
        )

        # scaler
        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        if noise_fix:
            self.likelihood.noise = 1e-4  # originally return self.noise_covar.noise
            self.likelihood.noise_covar.raw_noise.requires_grad = False

        self.follow_batch = follow_batch


    def forward(self, batch, **kwargs):
        x = self.transformer(batch)
        x = self.scale_to_bounds(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    def trainmodel(self, optimizer, scheduler, mll, epochs):
        for i in range(epochs):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            pred = self.likelihood(
                self.__call__(self.train_inputs))

            # Calc loss and backprop gradients
            loss = -mll(pred, self.train_targets)
            loss.backward()

            optimizer.step()
            scheduler.step()

class DKLCGCNN(ExactGP_graph):
    def __init__(self, train_x, train_y, likelihood, nn_param,
                 batch:Data, noise_fix=False):
        """
        :param train_x: dammy arg, should be None
        :param train_y: (N,) - Tensor
        :param params includes below dicts, key is '0', '1', '2' ....
            mpl_kernel_param: {'layers':List[int]}
            graph_param: {'n_layer':int, 'n_atom_fea':int, 'n_conv':int}
            pooling_param: {'layers':List[int,xx,yy,..,1]}
            embedding_param: {'n_atom_fea':int, 'num_atom':int, including dammy atom,
                             'dammy_indx':int, should be set as 0}
        :param batch: Data(graph)
        """

        super(DKLCGCNN, self).__init__(None, train_y.flatten(), likelihood)

        # transformer
        orig_atom_fea_len = nn_param['orig_atom_fea_len']
        nbr_fea_len = nn_param['nbr_fea_len']
        atom_fea_len = nn_param['atom_fea_len']
        n_conv = nn_param['n_conv']
        h_fea_len = nn_param['h_fea_len']
        n_h = nn_param['n_h']

        self.transformer = CrystalGraphConvNet(orig_atom_fea_len=orig_atom_fea_len,
                                               nbr_fea_len=nbr_fea_len,
                                               atom_fea_len=atom_fea_len,
                                               n_conv=n_conv,
                                               h_fea_len=h_fea_len,
                                               n_h=n_h)

        self.train_inputs = batch
        self.train_targets = train_y

        n_output_kernel = h_fea_len
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel=MaternKernel(
                nu=2.5,
                ard_num_dims=n_output_kernel,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
            outputscale_prior=GammaPrior(2.0, 0.15),
        )

        # scaler
        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        if noise_fix:
            # noise = torch.Tensor([1e-4])
            # self.likelihood.noise = torch.nn.Parameter(noise,requires_grad=True)
            self.likelihood.noise = 1e-4  # originally return self.noise_covar.noise
            self.likelihood.noise_covar.raw_noise.requires_grad = False

    def forward(self, batch):
        x = self.transformer(batch)
        x = self.scale_to_bounds(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def trainmodel(self,optimizer, scheduler, mll, epochs):
        for i in range(epochs):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            pred = self.likelihood(self.__call__(self.train_inputs))

            # Calc loss and backprop gradients
            loss = -mll(pred, self.train_targets)
            loss.backward()

            optimizer.step()
            scheduler.step()

class DKL(ExactGP_graph):
    def __init__(self, train_x,
                 train_y,
                 likelihood,
                 batch:Data,
                 transformer:torch.nn.Module,
                 mean_module:torch.nn.Module,
                 covar_module:torch.nn.Module,
                 noise_fix=False):
        """
        :param train_x: dammy arg, should be None
        :param train_y: (N,) - Tensor
        :param batch: Data(graph) (a replacement for train_x)
        """

        super(DKL, self).__init__(None, train_y.flatten(), likelihood,
                                  )

        self.transformer = transformer
        self.mean_module = mean_module
        self.covar_module = covar_module

        self.train_inputs = batch
        self.train_targets = train_y

        # scaler
        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        if noise_fix:
            self.likelihood.noise = 1e-4  # originally return self.noise_covar.noise
            self.likelihood.noise_covar.raw_noise.requires_grad = False

    def forward(self, batch:Data,**kwargs):
        x = self.transformer(batch)
        x = self.scale_to_bounds(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def trainmodel(self,optimizer, scheduler, mll, epochs):
        for i in range(epochs):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            pred = self.likelihood(self.__call__(self.train_inputs))

            # Calc loss and backprop gradients
            loss = -mll(pred, self.train_targets)
            loss.backward()

            optimizer.step()
            scheduler.step()

