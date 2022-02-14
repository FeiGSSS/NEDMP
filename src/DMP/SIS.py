# -*- encoding: utf-8 -*-
'''
@File    :   SIS.py
@Time    :   2021/05/21 15:57:18
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import torch

from torch_scatter import scatter
from copy import deepcopy
from src.DMP.utils import edgeList

class DMP_SIS():
    def __init__(self, weight_adj, nodes_p, max_steps=30):
        """
        This class implemented the rDMP equations for SIS model on graph.
        The details of this model refers to equations (2) (3) in paper:
             Shrestha, Munik, Samuel V. Scarpino, and Cristopher Moore. 
             "Message-passing approach for recurrent-state epidemic models 
             on networks." Physical Review E 92.2 (2015): 022821. 
        This implementation supports setting different parameters for edges and nodes.
        Parameters:
            weight_adj: np.array of size [N, N], storing the edge weight of each edge
            nodes_p: np.array of size [N], the probability for each infected nodes return to susceptible 
        """
        src_nodes, tar_nodes, edge_weight, cave_idx = edgeList(weight_adj)
        self.src_nodes = torch.LongTensor(src_nodes)
        self.tar_ndoes = torch.LongTensor(tar_nodes)
        self.edge_weight = torch.Tensor(edge_weight)
        self.cave_idx = torch.LongTensor(cave_idx)
        self.node_prob = torch.Tensor(nodes_p)
        self.node_prob_edge = self.node_prob[self.src_nodes]

        self.number_of_nodes = weight_adj.shape[0]
        self.number_of_edges = self.src_nodes.shape[0]

        self.max_steps = max_steps

        self.marginal_each_step = []

    def _set_seeds(self, seed_list):
        """
        setting the initial conditions using seed_list
        """
        # The probabilities being infectious and susceptible
        self.I = torch.zeros(self.number_of_nodes)
        self.S = torch.ones(self.number_of_nodes)
        for seed in seed_list:
            self.I[seed] = 1
            self.S[seed] = 0
        self.record()
        # self.message[i] is the message for edge [src_node[i], tar_node[i]]
        # If src_node[i] is seed node, then self.message[i] = 1, else 0
        self.message = torch.zeros(self.number_of_edges)
        for i, src in enumerate(self.src_nodes):
            if src in seed_list:
                self.message[i] = 1
    
    def record(self):
        """
        recording a [N, 2] tensor for each step
        """
        I = deepcopy(self.I)
        S = deepcopy(self.S)
        self.marginal_each_step.append(torch.stack((S, I), dim=1))

    def iteration(self):
        """
        One-step updating the message; Output the new I and S
        """
        message = self.message * self.edge_weight * self.S[self.tar_ndoes]
        message_aggregation = scatter(message, self.tar_ndoes, reduce="sum")

        # message update
        message_aggregation_cave = message_aggregation[self.src_nodes] - message[self.cave_idx]
        self.message = self.message - self.node_prob_edge * self.message + message_aggregation_cave

        # Nodes update
        self.I = self.I - self.node_prob * self.I + message_aggregation
        self.S = 1 - self.I

        return self.I, self.S

    def _stop(self):
        if len(self.marginal_each_step) < 2:
            return False
        else:
            former, later = self.marginal_each_step[-2:]
            delta = torch.max(torch.abs(former-later))
            if delta > 0.0001:
                return False
            else:
                return True
    
    def run(self, seed_list):
        assert isinstance(seed_list, list)
        seed_list = [int(seed) for seed in seed_list]
        self._set_seeds(seed_list)
        for step in range(self.max_steps):
            self.iteration()
            self.record()
            if self._stop():
                break
        # stack marginals for output
        marginals = torch.stack(self.marginal_each_step, dim=0) # ==> [T, N, 2]
        return marginals


        
        