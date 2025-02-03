import numpy as np
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from typing import Union, Any, Callable
import itertools
import re
import sys
import torch
import warnings
import os
import functools
import json
from torch.utils.data import Dataset
import pandas as pd
from torch_geometric.data import Data, DataLoader, Batch
from abc import abstractmethod

def standalize(x:Union[np.ndarray, torch.Tensor], mean = None, std = None,
               res_mask = None, reverse = False):
    if x.ndim == 1:
        x = x[:, None]
    if reverse:
        if mean is None:
            print('ERROR!!!')
            return
        else:
            return x * std + mean

    if mean is None:
        if isinstance(x, np.ndarray):
            mean = np.mean(x, axis=0)
            std = np.std(x, axis=0)
            res_mask = np.ones(len(mean), dtype=bool)
            res_mask[std == 0] = False
        elif isinstance(x, torch.Tensor):
            mean = torch.mean(x, dim=0)
            std = torch.std(x, dim=0)
            res_mask = torch.ones(len(mean), dtype=bool)
            res_mask[std == 0] = False
        else:
            print('ERROR!!!')
            return
        return (x[:, res_mask] - mean[res_mask]) / std[res_mask], [mean, std, res_mask]

    else:
        return (x[:, res_mask] - mean[res_mask]) / std[res_mask]


def normalize(x:Union[np.ndarray, torch.Tensor], max = None, min = None, res_mask = None, reverse=False):
    if x.ndim == 1:
        x = x[:, None]
    if reverse:
        if min is None:
            print('ERROR!!!')
            return
        else:
            return x * (max - min) + min

    if min is None:
        if isinstance(x, np.ndarray):
            max = np.max(x, axis=0)
            min = np.min(x, axis=0)
            res_mask = np.ones(len(max), dtype=bool)
            res_mask[(max - min) == 0] = False
        elif isinstance(x, torch.Tensor):
            max = torch.max(x, dim=0)
            min = torch.min(x, dim=0)
            res_mask = torch.ones(len(max), dtype=bool)
            res_mask[(max - min) == 0] = False
        else:
            print('ERROR!!!')
            return
        return (x[:, res_mask] - min[res_mask]) / (max[res_mask] - min[res_mask]), [max, min, res_mask]

    return (x[:, res_mask] - min[res_mask]) / (max[res_mask] - min[res_mask])


class SitesData(Data):
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if ('site' in key) and ('edge_index' in key):
            k = key.split('_')[-1]
            key_x = 'x_' + k
            return self.__getattr__(key_x).size(0)
        return super().__inc__(key, value, *args, **kwargs)

class ElementDescTable(object):
    def __init__(self, composition_columns, ABC, n_comb=2,
                 dtype = torch.float32, desc_func:Callable=None, prop_dict = None):
        """
        :param composition_columns:  List
        :param ABC: nested List, ex. [[Ar,Cu,Ta], [Ru,Sr,Ba,Te], [...]]
        :param dammy: location of dammy atoms, atomic number of dammy atoms is always "0"
                      name of dammy atom is "Dammy"
        :param n_comb: int or list
        """
        self.composition_columns = composition_columns
        self.ABC = ABC
        self.n_Site = len(ABC)
        if not hasattr(n_comb, '__iter__'):
            n_comb = [n_comb] * self.n_Site
        self.n_comb = n_comb
        self.dtype = dtype

        if desc_func is None:
            pass
        else:
            self.get_elem_desc = desc_func

        if prop_dict is None:
            prop_dict = {}
            norm_res = []
            for _ in range(self.n_Site):
                desc = []
                for elem in self.ABC[_]:
                    desc.append(self.get_elem_desc(elem))
                desc = np.array(desc)
                desc, tmp = standalize(desc)
                norm_res.append(tmp)
                for e, d  in zip(self.ABC[_],desc):
                    prop_dict[e] = d.tolist()
                char = chr(ord('a') + _)
                prop_dict['Dammy_' + char] = [0] * len(d)


            self.prop_dict = prop_dict
            self.norm_res = norm_res

        """ important variables and words """
        """
        ABC_site : [[Ar,Cu], [Ca,Sr,Ba],...] Nested List (double)
        ABC_sites : [[[Ar,Cu], [Ca,Sr,Ba],...], ..., [Au,Mg], [Ca,Sr,Ta]] Nested List (triple)
        xyz_site : [[0.8,0.2], [0.2,0.3,0.5],...] Nested List (double), index corresponds to that of ABC_site 
        xyz_sites : Similar to ABC_sites
        abc_site : [Ca,Sr,Ba]
        Site : each of A site, B site, C site, ...
        elem : Cu
        """

    @ abstractmethod
    def get_elem_desc(self,elem: str, **kwargs):
        """
        :param elem: str, name of the atom specie
        :return: List of element features
        """

        pass

    def get_comp_desc(self, ABC_site, xyz_site):
        desc = []
        for _, abc_site in enumerate(ABC_site):
            D = []
            for site in abc_site:
                d = self.prop_dict[site]
                D.append(d)
            D = np.array(D) * np.array(xyz_site[_])[:,np.newaxis]
            D = np.sum(D,axis=0)
            desc.append(D)
        desc = np.concatenate(desc,axis=0)

        return desc

    def site_to_table(self, abc_site: list, xyz_site: list):
        abc_site = [j for i in range(self.n_Site) for j in abc_site[i]]
        xyz_site = [j for i in range(self.n_Site) for j in xyz_site[i]]

        table_value = np.zeros(len(self.composition_columns))
        composition_columns = np.array(self.composition_columns)
        for site, ratio in zip(abc_site, xyz_site):
            table_value[composition_columns == site] = ratio

        return table_value

    def table_to_ABCxyzsite(self, table_value: Union[list, np.ndarray]):
        if isinstance(table_value, list):
            table_value = np.asarray(table_value)

        ABC_site = []
        xyz_site = []
        for abc in self.ABC:
            tmp1 = []
            tmp2 = []
            for site, val in zip(self.composition_columns, table_value):
                if val == 0:
                    continue
                if site in abc:
                    tmp1.append(site)
                    tmp2.append(val)
            ABC_site.append(tmp1)
            xyz_site.append(tmp2)

        return ABC_site, xyz_site

    def ABCxyzsite_to_table(self, ABC_sites, xyz_sites):
        if len(ABC_sites) != len(xyz_sites):
            print('Error!!')
            sys.exit(1)

        table = np.zeros((len(ABC_sites),len(self.composition_columns)))
        desc_columns = np.array(self.composition_columns)
        for i, ABC_site in enumerate(ABC_sites):
            for j, abc_site in enumerate(ABC_site):
                for k, site in enumerate(abc_site):
                    table[i,desc_columns == site] = xyz_sites[i][j][k]

        return table

    @staticmethod
    def check_duplicate_tables(table1: Union[list, np.ndarray],
                               table2 = None, indx = None):
        """
        When tabl2 is None, duplicates is checked in table1.

        :param table1: train
        :param table2: test or None
        :return:
        """
        if isinstance(table1, list):
            table1 = np.asarray(table1)
        if isinstance(table2, list):
            table2 = np.asarray(table2)

        dupl_indx = []  # duplicate indices in table2 (test data)
        if table2 is None:
            if indx is None:
                ValueError('indx is None')
            res, dupl_indx = np.unique(table1, axis=0, return_index=True)
            dupl_indx = np.setdiff1d(indx, dupl_indx)

            return dupl_indx

        for tb1 in table1:
            i = np.arange(len(table2))[np.all(np.isclose(table2, tb1), axis=1)]
            if len(i) != 0:
                dupl_indx.extend(i.tolist())
        return np.array(dupl_indx)

    def table_to_name(self, table_value, rd = 2):
        name = ''
        for site, val in zip(self.composition_columns, table_value):
            if val == 0:
                pass
            else:
                if val == 1:
                    name += '%s' % (site)
                else:
                    tmp =  site
                    name += tmp + str(round(val,rd))
        return name


    def get_ABCsites(self):
        comb = []
        for i in range(self.n_Site):
            tmp = []
            for r in range(1,self.n_comb[i] + 1):
                tmp.extend(list(itertools.combinations(self.ABC[i], r=r)))

            comb.append(tmp)
        comb = list(itertools.product(*comb))

        return comb

    def get_ABCsite_desc(self, ABC_site):
        desc = []
        for abc_site in ABC_site:
            tmp = []
            for site in abc_site:
                tmp.append(self.prop_dict[site])
            #tmp = np.asarray(tmp)
            desc.append(tmp)

        return desc

    def table_to_desc(self, table_value):
        ABC_sites, xyz_sites = self.table_to_ABCxyzsite(table_value)
        desc = []
        for abc in ABC_sites:
            tmp = []
            for site in abc:
                tmp.append(self.prop_dict[site])
            desc.append(tmp)

        return desc, ABC_sites, xyz_sites

    def generate_test_data(self, ratio_range=[0.0, 1.0], delta=0.05,
                           constrain=None, constrain_equal = True,
                           exclude_elem:Union[None,list] = None):
        if exclude_elem is None:
            exclude_elem = []
        for ABC_site in self.get_ABCsites():
            if bool(set(exclude_elem) & set([e2 for e1 in ABC_site for e2 in e1])):
                continue
            xyz_sites = []
            for sites in ABC_site:
                n = len(sites)
                x = np.arange(ratio_range[0], ratio_range[1] + delta, delta)
                x = x.tolist()
                x = [x for _ in range(n)]
                xyz = np.asarray(list(itertools.product(*x)))
                if constrain is None:
                    mask = np.ones(len(xyz), dtype=bool)
                else:
                    if constrain_equal:
                        mask = np.sum(xyz, axis=1) == constrain
                    else:
                        mask = np.sum(xyz, axis=1) <= constrain
                xyz = xyz[mask]
                xyz_sites.append(xyz.tolist())
            xyz_sites = list(itertools.product(*xyz_sites))
            ABC_sites = [ABC_site] * len(xyz_sites)

            yield ABC_sites, xyz_sites

    def table_to_weighted_desc(self, table_value):
        ABC_sites, xyz_sites = self.table_to_ABCxyzsite(table_value)
        desc = []
        for i, abc in enumerate(ABC_sites):
            tmp = []
            for j, site in enumerate(abc):
                d = self.prop_dict[site]
                d = np.array(d) * xyz_sites[i][j]
                tmp.append(d)
            tmp = np.sum(np.array(tmp),axis=0)
            desc.append(tmp)

        return desc

    def to_graph(self, table_value, y, graph_attr = None, site_attr = None):
        desc, ABC_site, xyz_site = self.table_to_desc(table_value)
        graphs = {}
        for i, abc in enumerate(ABC_site):
            edge_index = []
            node_fea = []
            ratio = []
            if len(abc) == 1:
                edge_index.append([0,0])
            elif len(abc) == 0:
                edge_index.append([0,0])
            else:
                _ = np.arange(0, len(abc))
                edge_index.extend(list(itertools.permutations(_, 2)))
            if len(abc) == 0:
                char = chr(ord('a') + i)
                node_fea.append(self.prop_dict['Dammy_' + char])
                ratio.append(0)
            else:
                for j, site in enumerate(abc):
                    node_fea.append(desc[i][j])
                    ratio.append(xyz_site[i][j])

            char = chr(ord('a')+i) # 0 to a, 1 to b, 2 to c ....
            key_x = 'x_' + char + 'site'
            key_edge_index ='edge_index_' + char + 'site'
            key_ratio ='ratio_' + char + 'site'
            graphs[key_x] = torch.tensor(node_fea, dtype=self.dtype)
            graphs[key_ratio] = torch.tensor(ratio, dtype=self.dtype)
            graphs[key_edge_index] = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            if site_attr is not None:
                graphs['site_attr_' + char + 'site'] = torch.tensor(site_attr[i], dtype=self.dtype)

        y = [y]
        data = SitesData.from_dict(graphs)
        data.update({'y':torch.tensor(y,dtype=self.dtype)})
        if graph_attr is not None:
            data.update({'graph_attr':torch.tensor(graph_attr, dtype=self.dtype)})

        return data, self.table_to_name(table_value)

    def collate_graphs_from_tablevalues(self, tablevalues, ys, follow_batch = None,
                                        graph_attr = None, site_attr = None):
        """
        :param tablevalues: (N, dim)
        :param ys: (N,)
        :param follow_batch: [x_asite, x_bsite, ...]
        :param graph_attr: (N,1,dim)
        :param site_attr: (N,n_abc,dim)
        :return:
        """
        graphs = []
        for _, (t, y) in enumerate(zip(tablevalues, ys)):
            ga = graph_attr[_] if graph_attr is not None else None
            sa = site_attr[_] if site_attr is not None else None
            data, name = self.to_graph(t, y, graph_attr = ga, site_attr = sa)
            graphs.append(data)
        graphs = Batch.from_data_list(graphs, follow_batch=follow_batch)

        return graphs

class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)

class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Parameters
    ----------

    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(self, root_dir, prop_col, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 ):
        self.root_dir = root_dir
        self.prop_col = prop_col
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            self.id_prop_data = pd.read_csv(f,header=0,index_col=0)
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx,dtype_float=torch.float,dtype_int=torch.int32):
        cif_id = self.id_prop_data.iloc[idx,0]
        target = self.id_prop_data.iloc[idx,self.prop_col]
        crystal = Structure.from_file(os.path.join(self.root_dir,
                                                   'id' + str(idx) +'.cif'))
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)

        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id

    def get_prop(self,idx):
        return self.id_prop_data.iloc[idx,self.prop_col].values


def myfeatualize(elem:str):
    def get_desc(elem):
        e = Element(elem)
        e_aff = e.electron_affinity  # electron affinity
        e_X = e.X  # electron negativity
        r_atom = e.data['Atomic radius']
        valence = e.electronic_structure.split('.')
        s_ele = 0
        p_ele = 0
        d_ele = 0
        f_ele = 0

        for v in valence:
            if bool(re.match('.[a-z][0-9]', v)):
                v = v[1:]
                v = list(v)
                if v[0] == 's':
                    s_ele = int(v[1])
                elif v[0] == 'p':
                    p_ele = int(v[1])
                elif v[0] == 'd':
                    d_ele = int(v[1])
                elif v[0] == 'f':
                    f_ele = int(v[1])
                else:
                    pass
            else:
                pass

        p = [e_aff,
             e_X,
             r_atom,
             s_ele,
             p_ele,
             d_ele,
             f_ele,
             ]
        return p

    if elem == 'MA': # methylammonium  (CH3NH3+)
        p = [get_desc(e) for e in ['C','H','H','H','N','H','H','H']]
        p = np.array(p)
        p = np.mean(p,axis=0)
    elif elem == 'FA': # formamidinium (NH2CHNH2+)
        p = [get_desc(e) for e in ['N','H','H','C','H','N','H','H']]
        p = np.array(p)
        p = np.mean(p,axis=0)

    else:
        p = np.array(get_desc(elem))
    return p
