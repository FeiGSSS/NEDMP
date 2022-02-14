# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2021/03/30 20:38:14
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
from numpy.lib.arraysetops import isin
from scipy import sparse
from scipy.sparse import block_diag
import networkx as nx
import scipy.sparse as sp
import numpy as np

import torch
from torch.utils.data import Dataset
import pickle as pkl


class graph_dataset(Dataset):
    def __init__(self, root, device, nedmp=False):
        super(graph_dataset, self).__init__()
        self.data = self.load(root)
        self.device = device
        self.nedmp = nedmp

        self.data["adj"] = [self.adj(g) for g in self.data["graph"]]
        self.data["simu_marginal"] = [x[1:] for x in self.data["simu_marginal"]] # remove t=0
        self.data["simu_marginal"] = [self.cut_redundancy(x) for x in self.data["simu_marginal"]]
        if isinstance(self.data["dmp_marginal"][0], np.ndarray):
            self.data["dmp_marginal"] = [self.cut_redundancy(x) for x in self.data["dmp_marginal"]]
        else:
            self.data["dmp_marginal"] = [self.cut_redundancy(x.numpy()) for x in self.data["dmp_marginal"]]
        
        self.data2cuda()

    def load(self, root):
        with open(root, "rb") as f:
            data = pkl.load(f)
        return data

    def __len__(self):
        return len(self.data["snode2edge"])

    def __getitem__(self, index):
        if  not self.nedmp:
            snode2edge = self.data["snode2edge"][index]
            tnode2edge = self.data["tnode2edge"][index]
            edge2tnode = self.data["edge2tnode"][index]
            nb_matrix  = self.data["nb_matrix"][index]
            edge_prob = self.data["edge_prob"][index]

            node_prob = self.data["node_prob"][index]
            node_seed = self.data["node_seed"][index]

            simu_marginal = self.data["simu_marginal"][index]
            dmp_marginal = self.data["dmp_marginal"][index]

            return (snode2edge, 
                    tnode2edge, 
                    edge2tnode, 
                    nb_matrix, 
                    edge_prob, 
                    node_prob, 
                    node_seed, 
                    simu_marginal, 
                    dmp_marginal)
        else:
            edge2tnode = self.data["edge2tnode"][index]
            nb_matrix  = self.data["nb_matrix"][index]
            cave_index = self.data["cave_index"][index]
            adj = self.data["adj"][index]
            
            weights = self.data["edge_prob"][index].squeeze()
            nodes_gamma = self.data["node_prob"][index].squeeze()
            node_seed = self.data["node_seed"][index]
            simu_marginal = self.data["simu_marginal"][index]
            dmp_marginal = self.data["dmp_marginal"][index]

            adj_index = adj.coalesce().indices()
            seed_list = torch.nonzero(node_seed.squeeze()).squeeze()

            return (edge2tnode, 
                    nb_matrix, 
                    adj_index, 
                    cave_index, 
                    weights, 
                    nodes_gamma, 
                    seed_list, 
                    simu_marginal, 
                    dmp_marginal)
    
    def adj(self, graph):
        """Return the sparse adj of graph"""
        N = graph.number_of_nodes()
        row, col = np.array(graph.edges()).T
        data = np.ones_like(row)
        adj = sparse.coo_matrix((data, (row, col)), shape=(N, N))
        return adj

    def data2cuda(self):
        for k, v in self.data.items():
            if not isinstance(v, list): continue
            if isinstance(v[0], sp.coo_matrix):
                self.data[k] = [self.sci2torch(x).to(self.device) for x in self.data[k]]
            elif isinstance(v[0], np.ndarray):
                if v[0].dtype == np.long:
                    self.data[k] = [torch.LongTensor(x).to(self.device) for x in self.data[k]]
                else:
                    self.data[k] = [torch.Tensor(x).to(self.device) for x in self.data[k]]
            else:
                continue
                
    def sci2torch(self, sparse_mat):
        values = sparse_mat.data
        indices = np.vstack((sparse_mat.row, sparse_mat.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = sparse_mat.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))
    
    def cut_redundancy(self, marginals):
        # cut redundancy from marginal
        for i in range(1, len(marginals)):
            delta = marginals[i-1, :,-1] - marginals[i,:,-1]
            delta = np.max(np.abs(delta))
            if delta<5E-3:
                break
        marginals = marginals[:i]
        return marginals
