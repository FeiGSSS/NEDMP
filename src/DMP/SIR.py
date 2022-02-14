# -*- encoding: utf-8 -*-
'''
@File    :   dmp_ic.py
@Time    :   2021/04/02 13:26:45
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''


from functools import reduce
import networkx as nx
import torch as T
from torch_scatter import scatter
from torch_geometric.utils import degree

from src.DMP.utils import edgeList

class DMP_SIR():
    def __init__(self, weight_adj, nodes_gamma): 
        self.edge_list = edgeList(weight_adj)
        # edge_list with size [3, E], (src_node, tar_node, weight) 
        self.src_nodes = T.LongTensor(self.edge_list[0])
        self.tar_nodes = T.LongTensor(self.edge_list[1])
        self.weights   = T.FloatTensor(self.edge_list[2])
        self.cave_index = T.LongTensor(self.edge_list[3])
        self.gamma = T.FloatTensor(nodes_gamma)[self.src_nodes]
        self.nodes_gamma = T.FloatTensor(nodes_gamma)
        
        self.N = max([T.max(self.src_nodes), T.max(self.tar_nodes)]).item()+1
        self.E = len(self.src_nodes)
        self.marginals = []


    def mulmul(self, Theta_t):
        Theta = scatter(Theta_t, index=self.tar_nodes, reduce="mul", dim_size=self.N) # [N]
        Theta = Theta[self.src_nodes] #[E]
        Theta_cav = scatter(Theta_t, index=self.cave_index, reduce="mul", dim_size=self.E+1)[:self.E]

        mul = Theta / Theta_cav
        return mul

    def _set_seeds(self, seed_list):
        self.seeds = T.zeros(self.N)
        self.seeds[seed_list] = 1

        # initial
        self.Ps_0 = 1 - self.seeds
        self.Pi_0 = self.seeds
        self.Pr_0 = T.zeros_like(self.seeds)

        self.Ps_i_0 = self.Ps_0[self.src_nodes]
        self.Pi_i_0 = self.Pi_0[self.src_nodes]
        self.Pr_i_0 = self.Pr_0[self.src_nodes]
        
        self.Phi_ij_0 = 1 - self.Ps_i_0
        self.Theta_ij_0 = T.ones(self.E)      

        # first iteration, t=1
        self.Theta_ij_t = self.Theta_ij_0 - self.weights * self.Phi_ij_0 + 1E-10 # get rid of NaN
        self.Ps_ij_t_1 = self.Ps_i_0 # t-1
        self.Ps_ij_t = self.Ps_i_0 * self.mulmul(self.Theta_ij_t) # t
        self.Phi_ij_t = (1-self.weights)*(1-self.gamma)*self.Phi_ij_0 - (self.Ps_ij_t-self.Ps_ij_t_1)

        # marginals
        self.Ps_t = self.Ps_0 * scatter(self.Theta_ij_t, self.tar_nodes, reduce="mul", dim_size=self.N)
        self.Pr_t = self.Pr_0 + self.nodes_gamma*self.Pi_0
        self.Pi_t = 1 - self.Ps_t - self.Pr_t
        self.marginals.append([self.Ps_t, self.Pi_t, self.Pr_t])

        # print(T.stack([self.Ps_t, self.Pi_t, self.Pr_t], dim=1))

    

    def iteration(self):
        self.Theta_ij_t = self.Theta_ij_t - self.weights * self.Phi_ij_t
        new_Ps_ij_t = self.Ps_i_0 * self.mulmul(self.Theta_ij_t)
        self.Ps_ij_t_1 = self.Ps_ij_t
        self.Ps_ij_t = new_Ps_ij_t
        self.Phi_ij_t = (1-self.weights)*(1-self.gamma)*self.Phi_ij_t - (self.Ps_ij_t-self.Ps_ij_t_1)

        # marginals
        self.Ps_t = self.Ps_0 * scatter(self.Theta_ij_t, self.tar_nodes, reduce="mul", dim_size=self.N)
        self.Pr_t = self.Pr_t + self.nodes_gamma*self.Pi_t
        self.Pi_t = 1 - self.Ps_t - self.Pr_t
        self.marginals.append([self.Ps_t, self.Pi_t, self.Pr_t])

        # print(T.stack([self.Ps_t, self.Pi_t, self.Pr_t], dim=1))


    def _stop(self):
        I_former, R_former = self.marginals[-2][1:]
        I_later , R_later  = self.marginals[-1][1:]

        I_delta = T.sum(T.abs(I_former-I_later))
        R_delta = T.sum(T.abs(R_former-R_later))
        if I_delta>0.01 or R_delta>0.01:
            return False
        else:
            return True

    def output(self):
        marginals = [T.stack(m, dim=1) for m in self.marginals]
        marginals = T.stack(marginals, dim=0) 
        return marginals

    def run(self, seed_list):
        self._set_seeds(seed_list)
        while True:
            self.iteration()
            if self._stop():
                break
        # Output a size of [T, N, 3] Tensor， T starts from t=1
        return self.output()
