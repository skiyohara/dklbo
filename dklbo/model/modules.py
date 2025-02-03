import numpy as np
import sys
import itertools
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import gpytorch
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.models.gp import GP
import warnings
from copy import deepcopy
from torch_scatter import scatter
from torch_geometric.data import Data
from gpytorch.models.exact_prediction_strategies import prediction_strategy
from torch_geometric.loader import DataLoader as DataLoader_geo
from torch_geometric.data import Batch, collate
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.utils.generic import length_safe_zip
from gpytorch import settings


class ExactGP_graph(GP):
    r"""
    The base class for any Gaussian process latent function to be used in conjunction
    with exact inference.

    :param torch.Tensor train_inputs: (size n x d) The training features :math:`\mathbf X`.
    :param torch.Tensor train_targets: (size n) The training targets :math:`\mathbf y`.
    :param ~gpytorch.likelihoods.GaussianLikelihood likelihood: The Gaussian likelihood that defines
        the observational distribution. Since we're using exact inference, the likelihood must be Gaussian.

    The :meth:`forward` function should describe how to compute the prior latent distribution
    on a given input. Typically, this will involve a mean and kernel function.
    The result must be a :obj:`~gpytorch.distributions.MultivariateNormal`.

    Calling this model will return the posterior of the latent Gaussian process when conditioned
    on the training data. The output will be a :obj:`~gpytorch.distributions.MultivariateNormal`.

    Example:
        >>> class MyGP(gpytorch.models.ExactGP):
        >>>     def __init__(self, train_x, train_y, likelihood):
        >>>         super().__init__(train_x, train_y, likelihood)
        >>>         self.mean_module = gpytorch.means.ZeroMean()
        >>>         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        >>>
        >>>     def forward(self, x):
        >>>         mean = self.mean_module(x)
        >>>         covar = self.covar_module(x)
        >>>         return gpytorch.distributions.MultivariateNormal(mean, covar)
        >>>
        >>> # train_x = ...; train_y = ...
        >>> likelihood = gpytorch.likelihoods.GaussianLikelihood()
        >>> model = MyGP(train_x, train_y, likelihood)
        >>>
        >>> # test_x = ...;
        >>> model(test_x)  # Returns the GP latent function at test_x
        >>> likelihood(model(test_x))  # Returns the (approximate) predictive posterior distribution at test_x
    """

    def __init__(self, train_inputs, train_targets, likelihood):
        if train_inputs is not None and torch.is_tensor(train_inputs):
            train_inputs = (train_inputs,)
        if train_inputs is not None and not all(torch.is_tensor(train_input) for train_input in train_inputs):
            raise RuntimeError("Train inputs must be a tensor, or a list/tuple of tensors")
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("ExactGP can only handle Gaussian likelihoods")

        super(ExactGP_graph, self).__init__()
        if train_inputs is not None:
            self.train_inputs = tuple(tri.unsqueeze(-1) if tri.ndimension() == 1 else tri for tri in train_inputs)
            self.train_targets = train_targets
        else:
            self.train_inputs = None
            self.train_targets = None
        self.likelihood = likelihood

        self.prediction_strategy = None

    @property
    def train_targets(self):
        return self._train_targets

    @train_targets.setter
    def train_targets(self, value):
        object.__setattr__(self, "_train_targets", value)

    def _apply(self, fn):
        if self.train_inputs is not None:
            for k, v in self.train_inputs.items():
                self.train_inputs[k] = fn(v)
            self.train_targets = fn(self.train_targets)
        return super(ExactGP_graph, self)._apply(fn)

    def _clear_cache(self):
        # The precomputed caches from test time live in prediction_strategy
        self.prediction_strategy = None

    def local_load_samples(self, samples_dict, memo, prefix):
        """
        Replace the model's learned hyperparameters with samples from a posterior distribution.
        """
        # Pyro always puts the samples in the first batch dimension
        num_samples = next(iter(samples_dict.values())).size(0)
        self.train_inputs = tuple(tri.unsqueeze(0).expand(num_samples, *tri.shape) for tri in self.train_inputs)
        self.train_targets = self.train_targets.unsqueeze(0).expand(num_samples, *self.train_targets.shape)
        super().local_load_samples(samples_dict, memo, prefix)

    def set_train_data(self, inputs=None, targets=None, strict=True):
        """
        Set training data (does not re-fit model hyper-parameters).

        :param torch.Tensor inputs: The new training inputs.
        :param torch.Tensor targets: The new training targets.
        :param bool strict: (default True) If `True`, the new inputs and
            targets must have the same shape, dtype, and device
            as the current inputs and targets. Otherwise, any shape/dtype/device are allowed.
        """
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            inputs = tuple(input_.unsqueeze(-1) if input_.ndimension() == 1 else input_ for input_ in inputs)
            if strict:
                for input_, t_input in length_safe_zip(inputs, self.train_inputs or (None,)):
                    for attr in {"shape", "dtype", "device"}:
                        expected_attr = getattr(t_input, attr, None)
                        found_attr = getattr(input_, attr, None)
                        if expected_attr != found_attr:
                            msg = "Cannot modify {attr} of inputs (expected {e_attr}, found {f_attr})."
                            msg = msg.format(attr=attr, e_attr=expected_attr, f_attr=found_attr)
                            raise RuntimeError(msg)
            self.train_inputs = inputs
        if targets is not None:
            if strict:
                for attr in {"shape", "dtype", "device"}:
                    expected_attr = getattr(self.train_targets, attr, None)
                    found_attr = getattr(targets, attr, None)
                    if expected_attr != found_attr:
                        msg = "Cannot modify {attr} of targets (expected {e_attr}, found {f_attr})."
                        msg = msg.format(attr=attr, e_attr=expected_attr, f_attr=found_attr)
                        raise RuntimeError(msg)
            self.train_targets = targets
        self.prediction_strategy = None

    def get_fantasy_model(self, inputs, targets, **kwargs):
        """
        Returns a new GP model that incorporates the specified inputs and targets as new training data.

        Using this method is more efficient than updating with `set_train_data` when the number of inputs is relatively
        small, because any computed test-time caches will be updated in linear time rather than computed from scratch.

        .. note::
            If `targets` is a batch (e.g. `b x m`), then the GP returned from this method will be a batch mode GP.
            If `inputs` is of the same (or lesser) dimension as `targets`, then it is assumed that the fantasy points
            are the same for each target batch.

        :param torch.Tensor inputs: (`b1 x ... x bk x m x d` or `f x b1 x ... x bk x m x d`) Locations of fantasy
            observations.
        :param torch.Tensor targets: (`b1 x ... x bk x m` or `f x b1 x ... x bk x m`) Labels of fantasy observations.
        :return: An `ExactGP` model with `n + m` training examples, where the `m` fantasy examples have been added
            and all test-time caches have been updated.
        :rtype: ~gpytorch.models.ExactGP
        """
        if self.prediction_strategy is None:
            raise RuntimeError(
                "Fantasy observations can only be added after making predictions with a model so that "
                "all test independent caches exist. Call the model on some data first!"
            )

        model_batch_shape = self.train_inputs[0].shape[:-2]

        if not isinstance(inputs, list):
            inputs = [inputs]

        inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in inputs]

        if not isinstance(self.prediction_strategy.train_prior_dist, MultitaskMultivariateNormal):
            data_dim_start = -1
        else:
            data_dim_start = -2

        target_batch_shape = targets.shape[:data_dim_start]
        input_batch_shape = inputs[0].shape[:-2]
        tbdim, ibdim = len(target_batch_shape), len(input_batch_shape)

        if not (tbdim == ibdim + 1 or tbdim == ibdim):
            raise RuntimeError(
                f"Unsupported batch shapes: The target batch shape ({target_batch_shape}) must have either the "
                f"same dimension as or one more dimension than the input batch shape ({input_batch_shape})"
            )

        # Check whether we can properly broadcast batch dimensions
        try:
            torch.broadcast_shapes(model_batch_shape, target_batch_shape)
        except RuntimeError:
            raise RuntimeError(
                f"Model batch shape ({model_batch_shape}) and target batch shape "
                f"({target_batch_shape}) are not broadcastable."
            )

        if len(model_batch_shape) > len(input_batch_shape):
            input_batch_shape = model_batch_shape
        if len(model_batch_shape) > len(target_batch_shape):
            target_batch_shape = model_batch_shape

        # If input has no fantasy batch dimension but target does, we can save memory and computation by not
        # computing the covariance for each element of the batch. Therefore we don't expand the inputs to the
        # size of the fantasy model here - this is done below, after the evaluation and fast fantasy update
        train_inputs = [tin.expand(input_batch_shape + tin.shape[-2:]) for tin in self.train_inputs]
        train_targets = self.train_targets.expand(target_batch_shape + self.train_targets.shape[data_dim_start:])

        full_inputs = [
            torch.cat(
                [train_input, input.expand(input_batch_shape + input.shape[-2:])],
                dim=-2,
            )
            for train_input, input in length_safe_zip(train_inputs, inputs)
        ]
        full_targets = torch.cat(
            [train_targets, targets.expand(target_batch_shape + targets.shape[data_dim_start:])], dim=data_dim_start
        )

        try:
            fantasy_kwargs = {"noise": kwargs.pop("noise")}
        except KeyError:
            fantasy_kwargs = {}

        full_output = super(ExactGP_graph, self).__call__(*full_inputs, **kwargs)

        # Copy model without copying training data or prediction strategy (since we'll overwrite those)
        old_pred_strat = self.prediction_strategy
        old_train_inputs = self.train_inputs
        old_train_targets = self.train_targets
        old_likelihood = self.likelihood
        self.prediction_strategy = None
        self.train_inputs = None
        self.train_targets = None
        self.likelihood = None
        new_model = deepcopy(self)
        self.prediction_strategy = old_pred_strat
        self.train_inputs = old_train_inputs
        self.train_targets = old_train_targets
        self.likelihood = old_likelihood

        new_model.likelihood = old_likelihood.get_fantasy_likelihood(**fantasy_kwargs)
        new_model.prediction_strategy = old_pred_strat.get_fantasy_strategy(
            inputs, targets, full_inputs, full_targets, full_output, **fantasy_kwargs
        )

        # if the fantasies are at the same points, we need to expand the inputs for the new model
        if tbdim == ibdim + 1:
            new_model.train_inputs = [fi.expand(target_batch_shape + fi.shape[-2:]) for fi in full_inputs]
        else:
            new_model.train_inputs = full_inputs
        new_model.train_targets = full_targets

        return new_model

    def __call__(self, *args, **kwargs):
        train_inputs = self.train_inputs
        inputs = args[0]

        # Training mode: optimizing
        if self.training:
            if self.train_inputs is None:
                raise RuntimeError(
                    "train_inputs, train_targets cannot be None in training mode. "
                    "Call .eval() for prior predictions, or call .set_train_data() to add training data."
                )
            res = super().__call__(inputs, **kwargs)
            return res

        # Prior mode
        elif settings.prior_mode.on() or self.train_inputs is None or self.train_targets is None:
            full_inputs = args
            full_output = super(ExactGP_graph, self).__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            return full_output

        # Posterior mode
        else:
            # Get the terms that only depend on training data
            if self.prediction_strategy is None:
                train_output = super().__call__(train_inputs, **kwargs)

                # Create the prediction strategy for
                self.prediction_strategy = prediction_strategy(
                    train_inputs=train_inputs,
                    train_prior_dist=train_output,
                    train_labels=self.train_targets,
                    likelihood=self.likelihood,
                )

            # Concatenate the input to the training input
            if kwargs == {}:
                full_inputs = Batch.from_data_list(train_inputs.to_data_list() + inputs.to_data_list())
            else:
                full_inputs = Batch.from_data_list(train_inputs.to_data_list() + inputs.to_data_list(),
                                                   follow_batch=kwargs['follow_batch'])
            # Get the joint distribution for training/test data
            full_output = super(ExactGP_graph, self).__call__(full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix


            # Determine the shape of the joint distribution
            batch_shape = full_output.batch_shape
            joint_shape = full_output.event_shape
            tasks_shape = joint_shape[1:]  # For multitask learning
            test_shape = torch.Size([joint_shape[0] - self.prediction_strategy.train_shape[0], *tasks_shape])

            # Make the prediction
            with settings.cg_tolerance(settings.eval_cg_tolerance.value()):
                (
                    predictive_mean,
                    predictive_covar,
                ) = self.prediction_strategy.exact_prediction(full_mean, full_covar)

            # Reshape predictive mean to match the appropriate event shape
            predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()
            return full_output.__class__(predictive_mean, predictive_covar)




class Transformer(nn.Module):
    def __init__(self, atom_fea_len, num_head = 3, drop_out = 0.1):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(Transformer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.num_head = num_head

        self.linear_Q = nn.Linear(atom_fea_len, atom_fea_len * num_head, bias=False)
        self.linear_K = nn.Linear(atom_fea_len, atom_fea_len * num_head, bias=False)
        self.linear_V = nn.Linear(atom_fea_len, atom_fea_len * num_head, bias=False)
        self.linear = nn.Linear(atom_fea_len * num_head, atom_fea_len, bias = False)
        self.softmax = nn.Softmax(dim = 3)
        self.dropout = nn.Dropout(drop_out)

    def split_head(self, x):
        x = torch.tensor_split(x, self.num_head, dim = -1)
        x = torch.stack(x, dim=1)
        return

    def concat_head(self, x):
        x = torch.tensor_split(x, self.num_head, dim = 1)
        x = torch.concat(x, dim = 3).squeeze(dim = 1)
        return x


    def forward(self, Q, K, V, mask=None):
        Q = self.linear_Q(Q)  # (BATCH_SIZE, max_num_elements, atom_fea * num_head)
        K = self.linear_K(K)
        V = self.linear_V(V)

        Q = self.split_head(Q)  # (BATCH_SIZE, head_num, max_num_elements, atom_fea)
        K = self.split_head(K)
        V = self.split_head(V)

        """
        softmax(Q*transpose(K)/root(dim))*V
        """
        QK = torch.matmul(Q, torch.transpose(K, 3, 2)) # Q * transpose(X)
        QK = QK / ((self.atom_fea_len) ** 0.5)

        if mask is not None:
            QK = QK + mask

        softmax_QK = self.softmax(QK)
        softmax_QK = self.dropout(softmax_QK)

        QKV = torch.matmul(softmax_QK, V)
        QKV = self.concat_head(QKV)
        QKV = self.linear(QKV)

        return QKV

class Attention(nn.Module):
    """
    Initialize ConvLayer.

    Parameters
    ----------

    atom_fea_len: int
      Number of atom hidden features.
    nbr_fea_len: int
      Number of bond features.
    """
    def __init__(self, ndim):
        super(Attention, self).__init__()

        self.linear = nn.Linear(ndim, ndim)
        self.softmax = nn.Softmax(dim=ndim)

    def forward(self, x):
        return x * self.softmax(self.linear(x))




