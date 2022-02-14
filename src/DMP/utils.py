# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2021/04/29 09:54:58
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import numpy as np
import networkx as nx
import scipy.sparse as sp

def cave_index(src_nodes, tar_nodes):
    edge_list = [(int(s), int(t)) for s, t in zip(src_nodes, tar_nodes)]
    E = len(edge_list)
    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    attr = {edge:w for edge, w in zip(edge_list, range(E))}
    nx.set_edge_attributes(G, attr, "idx")

    cave = []
    for edge in edge_list:
        if G.has_edge(*edge[::-1]):
            cave.append(G.edges[edge[::-1]]["idx"])
        else:
            cave.append(E)
    return cave

def edgeList_ic(weight_adj):
    """From weighted adj generate a [4, 2*E] edge list. edge_list[2] is the directed edge weight  

    Args:
        weight_adj (np.array): weighted adj

    Returns:
        [type]: [description]
    """
    sp_mat = sp.coo_matrix(weight_adj)
    weight = sp_mat.data
    cave = cave_index(sp_mat.row, sp_mat.col)
    edge_list = np.vstack((sp_mat.row, sp_mat.col, weight, cave))
    return edge_list

def edgeList(weight_adj):
    sp_mat = sp.coo_matrix(weight_adj)
    weight = sp_mat.data
    cave = cave_index(sp_mat.row, sp_mat.col)
    edge_list = np.vstack((sp_mat.row, sp_mat.col, weight, cave))
    return edge_list