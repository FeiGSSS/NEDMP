from operator import sub
import os
import sys

from networkx.algorithms.shortest_paths import dense
from pathlib import Path
import sys
sys.path.append(str(Path("..").parent.absolute()))
import argparse
import time
import subprocess
from tqdm import tqdm

from copy import deepcopy
from src.utils.simulation import SIMU, SIR_ndlib
from src.utils.utils import line_graph
from src.DMP.SIR import DMP_SIR

import pickle as pkl
import networkx as nx
import numpy as np

def cave_index_fun(src_nodes, tar_nodes):
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
    return np.array(cave)



if __name__ == "__main__":

    NUM_TEST = 100

    data_path = ["../data/origin_small/dense_ER/mu05",
                 "../data/origin_small/dense_ER/mu10",
                 "../data/origin_small/dense_Regular/mu05", 
                 "../data/origin_small/dense_Regular/mu10",
                 "../data/origin_small/ER/mu05",
                 "../data/origin_small/ER/mu10",
                 "../data/origin_small/Regular/mu05", 
                 "../data/origin_small/Regular/mu10"]

    data_path = [os.path.join(dp, "test_data") for dp in data_path]
    for dp in data_path:
        if not os.path.exists(dp):
            os.mkdir(dp)
    
    lambda_value = np.linspace(0.05, 0.95, 20)

    for cnt, dp in enumerate(data_path):
        sub_dp  = [os.path.join(dp, "lambda{}.pkl".format(i)) for i in range(20)]
        if "dense" in dp:
            lambda_value = np.linspace(0.05, 0.5, 20)
        else:
            lambda_value = np.linspace(0.05, 0.95, 20)

        for lv, sp in zip(lambda_value, sub_dp):
            dataset = {"seeds": [],
                    "snode2edge":[],
                    "tnode2edge":[],
                    "edge2tnode":[],
                    "nb_matrix":[],
                    "node_prob": [],
                    "edge_prob":[],
                    "node_seed":[],
                    "cave_index":[],
                    "simu_marginal":[]}
            # generate graph
            if "dense" in sp:
                #  平均度=10
                k = 10
                n = 100
                if "ER" in sp:
                    graph = nx.fast_gnp_random_graph(n=n, p=2*k/n)
                elif "Regular" in sp:
                    graph = nx.random_regular_graph(d=k, n=n)
            else:
                #  平均度=4
                k = 4
                n = 100
                if "ER" in sp:
                    graph = nx.fast_gnp_random_graph(n=n, p=2*k/n)
                elif "Regular" in sp:
                    graph = nx.random_regular_graph(d=k, n=n)
  
            # rename nodes
            node_rename = {n:i for i, n in enumerate(graph.nodes())}
            graph = nx.relabel_nodes(graph, node_rename)
            graph = nx.DiGraph(graph)
            # generate transformation matrix
            edge_index = np.array(list(graph.edges()))
            N = graph.number_of_nodes()
            E = graph.number_of_edges()
            snode2edge, tnode2edge, edge2tnode, NB_matrix = line_graph(edge_index, N, E)
            row, col = np.array(graph.edges()).T
            cave_index = cave_index_fun(row, col)
            # save
            dataset["node2edge"] = snode2edge
            dataset["snode2edge"] = snode2edge
            dataset["tnode2edge"] = tnode2edge
            dataset["edge2tnode"] = edge2tnode
            dataset["nb_matrix"] = NB_matrix
            dataset["cave_index"] = cave_index

            # seeds and save
            seeds = np.random.choice(list(graph.nodes()), size=NUM_TEST, replace=True)
            dataset["seeds"] = seeds

            # set epidemic parameters
            if "mu05" in sp:
                node_prob = np.ones(N) * 0.5
            elif "mu10" in sp:
                node_prob = np.ones(N)
            else:
                raise ValueError
            weight = np.ones(E) * lv
            # save epidemic parameters
            node_prob_map = {n:p for n, p in zip(graph.nodes, node_prob)}
            nx.set_node_attributes(graph, node_prob_map, "node_prob")
            weight_map = {e:w for e, w in zip(graph.edges, weight)}
            nx.set_edge_attributes(graph, weight_map, "weight")
            dataset["node_prob"] = node_prob[:, None]
            edge_feat = np.array([weight_map[tuple(e)] for e in edge_index])[:, None] # size [2E, 1]
            dataset["edge_prob"] = edge_feat
            dataset["graph"] = graph

            # iterate different seeds
            for s in tqdm(seeds):
                node_seed = np.zeros(N)
                node_seed[s] = 1
                dataset["node_seed"].append(node_seed[:, None])

                simu_marginal = SIR_ndlib(graph, 
                                     beta=weight[0], 
                                     gamma=node_prob[0],
                                     seeds = [s],
                                     num_iterations=11)
                dataset["simu_marginal"].append(simu_marginal)

            with open(sp, "wb") as f:
                pkl.dump(dataset, f)
            print(sp, " DONE")
