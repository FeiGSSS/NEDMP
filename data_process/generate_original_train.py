import os
import sys
sys.path.append("..")
import argparse
import time
from tqdm import tqdm

from copy import deepcopy
from src.utils.simulation import SIMU
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-ns", "--num_samples", type=int, default=400,
                        help="number of samples to be generated")
    parser.add_argument("--dense",  action="store_true")
    args = parser.parse_args()
    
    if args.dense:
        data_path = ["../data/origin_small/dense_ER/mu05",
                     "../data/origin_small/dense_ER/mu10",
                     "../data/origin_small/dense_Regular/mu05", 
                     "../data/origin_small/dense_Regular/mu10"]
    else:
        data_path = ["../data/origin_small/ER/mu05",
                     "../data/origin_small/ER/mu10",
                     "../data/origin_small/Regular/mu05", 
                     "../data/origin_small/Regular/mu10"]

    save_path = [os.path.join(path, "train_data") for path in data_path]
    save_name = "{}_{}.pkl".format("SIR", args.num_samples)

    for dp in data_path:
        if not os.path.exists(dp):
            os.mkdir(dp)
    for sp in save_path:
        if not os.path.exists(sp):
            os.mkdir(sp)
    
    for sp in save_path:
        print(sp)
        dataset = {"config":args,
                "graph":[],
                "seeds": [],
                "snode2edge":[],
                "tnode2edge":[],
                "edge2tnode":[],
                "nb_matrix":[],
                "node_prob": [],
                "edge_prob":[],
                "node_seed":[],
                "cave_index":[],
                "simu_marginal":[],
                "dmp_marginal":[]}
        if args.dense:
            #  平均度=10
            k = 10
            n = 100
            if "ER" in sp:
                graph = nx.fast_gnp_random_graph(n=n, p=2*k/n)
            elif "Regular" in sp:
                graph = nx.random_regular_graph(d=k, n=n)
        elif not args.dense:
            #  平均度=4
            k = 4
            n = 100
            if "ER" in sp:
                graph = nx.fast_gnp_random_graph(n=n, p=2*k/n)
            elif "Regular" in sp:
                graph = nx.random_regular_graph(d=k, n=n)

        node_rename = {n:i for i, n in enumerate(graph.nodes())}
        graph = nx.relabel_nodes(graph, node_rename)

        graph = nx.DiGraph(graph) # Bi directed graph from original undirected graph
        # transformation matrix
        edge_index = np.array(list(graph.edges()))
        N = graph.number_of_nodes()
        E = graph.number_of_edges()
        snode2edge, tnode2edge, edge2tnode, NB_matrix = line_graph(edge_index, N, E)
        row, col = np.array(graph.edges()).T
        cave_index = cave_index_fun(row, col)
        

        dataset["node2edge"] = [snode2edge]*args.num_samples
        dataset["snode2edge"] = [snode2edge]*args.num_samples
        dataset["tnode2edge"] = [tnode2edge]*args.num_samples
        dataset["edge2tnode"] = [edge2tnode]*args.num_samples
        dataset["nb_matrix"] = [NB_matrix]*args.num_samples
        dataset["cave_index"] = [cave_index]*args.num_samples
        
        if "mu05" in sp:
            node_prob = np.ones(N) * 0.5
        elif "mu10" in sp:
            node_prob = np.ones(N)
        else:
            raise ValueError
        # seeds size
        seeds_size = 1

        for s in range(args.num_samples):
            t0 = time.time()
            G = deepcopy(graph)

            
            if not args.dense:
                # random weights from 0-1
                weight = np.ones(E) * np.random.rand()
            else:
                # random weights from 0-0.5
                weight = np.ones(E) * np.random.uniform(0, 0.5, 1)[0]

            # Setting parameters for edges nodes
            weight_map = {e:w for e, w in zip(G.edges, weight)}
            nx.set_edge_attributes(G, weight_map, "weight")

            node_prob_map = {n:p for n, p in zip(G.nodes, node_prob)}
            nx.set_node_attributes(G, node_prob_map, "node_prob")

            # Saving Graph with attributes
            dataset["graph"].append(G)

            # Generate info. for LGNN model
            dataset["node_prob"].append(node_prob[:, None])
            edge_feat = np.array([weight_map[tuple(e)] for e in edge_index])[:, None] # size [2E, 1]
            dataset["edge_prob"].append(edge_feat)

            seeds = np.random.choice(list(G.nodes()), size=seeds_size, replace=False)
            dataset["seeds"].append(seeds)

            # Seeds as feature
            node_seed = np.zeros(N)
            node_seed[seeds] = 1
            dataset["node_seed"].append(node_seed[:, None])

            # Simulation
            simu_marginal = SIMU("SIR", G, seeds, path="../src/utils/")
            dataset["simu_marginal"].append(simu_marginal)

            # DMP
            wadj = nx.adj_matrix(G).toarray()
            dmp_sir = DMP_SIR(weight_adj=wadj, nodes_gamma=node_prob)
            dmp_simulation = dmp_sir.run(list(seeds))
            dataset["dmp_marginal"].append(dmp_simulation)

            print("{:^3} Sample Generated, Time={:.1f}s, Influence={:.1f}/{}".\
                format(s, time.time()-t0, simu_marginal[-1, :, -1].sum(), G.number_of_nodes()))

        # Saving
        with open(os.path.join(sp, save_name), "wb") as f:
            pkl.dump(dataset, f)
