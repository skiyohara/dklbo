import torch
import torch.nn as nn
from torch_scatter import scatter



class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len,
                                 2 * self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        atom_fea = atom_in_fea[nbr_fea_idx] # -> (2,num_edges,atom_fea_len)
        total_nbr_fea = torch.cat([atom_fea[0],atom_fea[1],nbr_fea], dim=1)  # -> (num_edges,atom_fea_len*2+nbr_fea_len)
        total_gated_fea = self.bn1(self.fc_full(total_nbr_fea))
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=1)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = scatter(nbr_filter * nbr_core, nbr_fea_idx[0],dim=0) / torch.sqrt(torch.bincount(nbr_fea_idx[0])[:, None])
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)

        return out

class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """

    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=32, n_conv=3, h_fea_len=32, n_h=1,
                 ):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                              nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h - 1)])
        self.fc_out = nn.Linear(h_fea_len, 1)
        self.crys_out = nn.Linear(h_fea_len, h_fea_len)

        self.attention = nn.Linear(atom_fea_len, 1)

    def pooling_attention(self, atom_fea, crystal_atom_idx):
        N = crystal_atom_idx[-1] + 1

        summed_fea = [torch.sum(atom_fea[crystal_atom_idx==cai] * \
                                nn.functional.softmax(self.attention(atom_fea[crystal_atom_idx==cai]), dim=0),
                                dim=0,keepdim=True)
                      for cai in range(N)]
        return torch.cat(summed_fea, dim=0)

    def forward(self, batch):
        """
        Forward pass

        N_nodes: Total number of atoms in the batch
        N_edges: Total number of edges in the batch

        ----------

        atom_fea: Variable(torch.Tensor) shape (N_nodes, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N_edges, nbr_fea_len)
        nbr_fea_idx: torch.LongTensor shape (2, N_edges)
        crystal_atom_idx: torch.LongTensor shape (N_nodes,)
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

                atom_fea: torch.Tensor,
                nbr_fea: torch.Tensor,
                nbr_fea_idx: torch.Tensor,
                crystal_atom_idx: torch.Tensor):
        """

        atom_fea = batch.x
        nbr_fea = batch.edge_attr
        nbr_fea_idx = batch.edge_index
        crystal_atom_idx = batch.batch

        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling_attention(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)

        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))

        crys_fea = self.crys_out(crys_fea)

        return crys_fea


