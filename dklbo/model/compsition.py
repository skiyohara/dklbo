import torch
import torch.nn as nn
from torch_scatter import scatter


class CompGraphConvNet(nn.Module):
    def __init__(self,
                 n_abc_site:int,
                 n_conv:int,
                 layers_atom:list,
                 layers_conv:list,
                 layers_fc:list,
                 layers_pooling:list,
                 activation = nn.Mish(),
                 site_respective = True):
        """
        :param n_abc_site : Number of sites, e.g, AB => 2
        :param n_conv : Number of convolutions
        :param layers_atom : atom transfromation layer [in, h, h, .., out]
        :param layers_conv : convolution layers, nested list [[in1, h1, h1, .., out1], [in2, h2, h2, .., out2],...]
        :param layers_pooling : pooling layers [in, h, h, .., out]
        :param layers_fc : fully connected layers [in, h, h, .., out]
        :param activation : activation function
        :param site_respective : whether to construct convolution layers respective to each site
        """

        super(CompGraphConvNet, self).__init__()

        self.n_conv = n_conv
        if site_respective:
            self.n_abc_site = n_abc_site
        else:
            self.n_abc_site = 1
        self.site_respective = site_respective

        fc = []
        if len(layers_atom) == 0:
            self.atom_fc = nn.Identity()
        else:
            for _, (_1, _2) in enumerate(zip(layers_atom[0:-1],layers_atom[1:])):
                if _ == len(layers_atom) - 2:
                    fc.append(nn.Linear(_1, _2)) # plus 1 is for ratio
                else:
                    fc.append(nn.Linear(_1, _2))
                    fc.append(activation)
                    fc.append(nn.BatchNorm1d(_2))
            self.atom_fc = nn.Sequential(*fc)

        conv = []
        for _1 in range(self.n_abc_site):
            tmp = []
            for _2 in range(n_conv):
                tmp.append(SiteConv(layers_conv[_1][_2], activation=activation))
            conv.append(nn.ModuleList(tmp))
        self.conv = nn.ModuleList(conv)

        fcs = []
        for _ in range(self.n_abc_site):
            layers = layers_pooling[_]
            fc = []
            for _, (_1, _2) in enumerate(zip(layers[0:-1],layers[1:])):
                if _ == len(layers) - 2:
                    fc.append(nn.Linear(_1 + 1, _2)) # plus 1 is for ratio, _2 should be 1
                else:
                    fc.append(nn.Linear(_1 + 1, _2 + 1))
                    fc.append(activation)
            fcs.append(nn.Sequential(*fc))
        self.pools = nn.ModuleList(fcs)

        fc = []
        for _, (_1, _2) in enumerate(zip(layers_fc[0:-1],layers_fc[1:])):
            if _ == len(layers_fc) - 2:
                fc.append(nn.Linear(_1, _2)) # plus 1 is for ratio
            else:
                fc.append(nn.Linear(_1, _2))
                fc.append(activation)
                fc.append(nn.BatchNorm1d(_2))
        self.fc = nn.Sequential(*fc)
        self.bn_fc = nn.BatchNorm1d(layers_fc[0])

        self.out = nn.Linear(layers_fc[-1],1)

    def forward(self, batch):
        """

        :param batch:torch_geometric.data.Data
               keys are listed below
                    x_{a,b,c...}site : (N_site, n_node_fea)
                    edge_index_{a,b,c...}site : (2,N_edges)
                    ratio_{a,b,c...}site : (N_site,)
                    y : (N_mat,)
        :return:
        """

        # convolution
        d = []
        for _1 in range(self.n_abc_site):
            char = chr(ord('a')+_1)
            key_x = 'x_' + char + 'site'
            key_edge_index ='edge_index_' + char + 'site'
            key_ratio ='ratio_' + char + 'site'
            key_batch = key_x + '_batch'

            x = batch.__getattr__(key_x)
            edge_index = batch.__getattr__(key_edge_index)
            ratio = batch.__getattr__(key_ratio)[:, None]
            b = batch.__getattr__(key_batch)

            for _2 in range(len(self.conv[_1])):
                if _2 == 0:
                    x = self.atom_fc(x)
                x = self.conv[_1][_2]([x, edge_index, ratio])

            x = self.pooling([x, ratio, b, _1])
            d.append(x)

        d = torch.cat(d, dim = -1) # -> (N_site, n_node_fea * n_abc_site)
        d = self.bn_fc(d)

        # fully connected layer
        out = self.fc(d)

        return out


    def pooling(self, x:list):
        x, ratio, b, _ = x
        alpha = self.pools[_](torch.cat([x,ratio],dim=-1))
        alpha = torch.exp(alpha)  # -> (N_atoms, 1)
        alpha_sum = scatter(alpha, b, dim = 0)[b]
        alpha = alpha / alpha_sum
        x = x * alpha

        return scatter(x, b, dim=0, reduce='sum')

    def forward_prop(self, batch):
        out = self.forward(batch)

        return self.out(out)

class SiteConv(nn.Module):
    def __init__(self, layers, activation = nn.Mish(), p_ratio = 0.0):
        """

        :param Asites_desc: (N_Asite, n_desc)
        :param layers:
        :param activation:
        """

        super(SiteConv, self).__init__()

        # self-interaction part
        sc = []
        for _, (_1, _2) in enumerate(zip(layers[0:-1],layers[1:])):
            if _ == len(layers) - 2:
                sc.append(nn.Linear(_1 + 1, _2))
            else:
                sc.append(nn.Linear(_1 + 1, _2 + 1)) # adding 1 is for ratio
                sc.append(activation)
                sc.append(nn.BatchNorm1d(_2 + 1))
                sc.append(nn.Dropout(p=p_ratio))

        self.sc = nn.Sequential(*sc)

        # convolution part
        fc = []
        for _, (_1, _2) in enumerate(zip(layers[0:-1],layers[1:])):
            if _ == len(layers) - 2:
                fc.append(nn.Linear(_1 * 2 + 2, _2))
            else:
                fc.append(nn.Linear(_1 * 2 + 2, _2 * 2 + 2)) # multiplying 2 is for two elements, adding 2 is for ratio
                fc.append(activation)
                fc.append(nn.BatchNorm1d(_2 * 2 + 2))
                fc.append(nn.Dropout(p=p_ratio))

        self.fc = nn.Sequential(*fc)

    def forward(self, x:list):
        """

        :param x list including three inptus
                d,
                edge_index
                ratio
                batch
        :return:
        """

        d, edge_index, ratio = x

        # self interaction part
        d_self = self.sc(torch.cat([d,ratio],dim = -1)) # -> (N_sites, n_node_fea)

        # convolution part
        d = d[edge_index] # -> (2, N_edges, n_node_fea)
        ratio = ratio[edge_index] # -> (2, N_edges, 1)
        d = torch.cat([d[0], d[1], ratio[0],ratio[1]],dim=-1) # -> (N_edges, n_node_fea*2 + 2)
        d = self.fc(d) # -> (N_edges, n_node_fea)

        d = scatter(d, edge_index[0], dim=0, reduce='sum') / torch.sqrt(torch.bincount(edge_index[0])[:, None])


        return d_self + d

