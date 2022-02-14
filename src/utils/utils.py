# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2021/04/01 17:04:49
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''

import scipy.sparse as sp
import numpy as np
import torch
import networkx as nx

def line_graph(edge_index, N, E):
    """generate the line graph from edge_index

    Args:
        edge_index ([type]): [E, 2]
        N ([type]): number of nodes
        E ([type]): number of directed edges

    Returns:
        [type]: [description]
    """
    snode2edge = np.zeros((N, E))
    tnode2edge = np.zeros((N, E))
    edge2tnode  = np.zeros((E, N))
    NB_matrix  = np.zeros((E, E))
    assert edge_index.shape[0] == E
    assert np.max(edge_index) == N-1, "{} N-1={}".format(np.max(edge_index), N-1)

    puppet_graph = nx.DiGraph()
    puppet_graph.add_edges_from(edge_index)
    nx.set_edge_attributes(puppet_graph, {tuple(edge):cnt for cnt, edge in enumerate(edge_index)}, "cnt")
    for cnt, edge in enumerate(edge_index):
        snode, tnode = edge
        snode2edge[snode][cnt] = 1
        tnode2edge[tnode][cnt] = 1
        edge2tnode[cnt][tnode] = 1
        for tnode2 in puppet_graph.successors(tnode):
            if snode != tnode2:
                cnt2 = puppet_graph[tnode][tnode2]["cnt"]
                NB_matrix[cnt][cnt2] = 1

    return sp.coo_matrix(snode2edge.T), \
           sp.coo_matrix(tnode2edge.T), \
           sp.coo_matrix(edge2tnode.T), \
           sp.coo_matrix(NB_matrix.T)

def aligning(target, x):
    """align two tensor by time dimension

    Args:
        target (target tensor): of size [T1, N, K]
        x ([type]): of size [T, N, K]

    Returns:
        aligned x: of size [T1, N, K]
    """
    step = target.shape[0]
    xstep = x.shape[0]
    if xstep>=step:
        x = x[:step]
    else:
        tail = x[-1].squeeze()
        padd = torch.stack([tail]*(step-xstep), dim=0)
        x = torch.cat([x, padd], dim=0)
    return x

def L1_error(pred_list, label_list):
    """
    pred_list: a list of [T, N, K] array
    label_list: same as pred_list
    """
    pred = np.vstack([x.reshape(-1, 1) for x in pred_list])
    label = np.vstack([x.reshape(-1, 1) for x in label_list])

    l1_error = np.mean(np.abs(pred-label))
    l1_error_std = np.std(np.abs(pred-label))
    return l1_error, l1_error_std