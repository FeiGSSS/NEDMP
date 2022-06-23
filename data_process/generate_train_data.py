from pathlib import Path
import sys
sys.path.append(str(Path("..").parent.absolute()))

import os
import argparse
import time

from copy import deepcopy
from src.utils.simulation import SIMU
from src.utils.utils import line_graph
from src.DMP.SIR import DMP_SIR
from src.DMP.SIS import DMP_SIS

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

###############################################

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
        padd = np.stack([tail]*(step-xstep), axis=0)
        x = np.concatenate([x, padd], axis=0)
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
    return l1_error

###############################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", "--data_path", type=str, 
                        help="path to graphs")
    parser.add_argument("-df", "--diffusion", type=str, default="SIR",
                        help="The diffusion model")
    parser.add_argument("-ns", "--num_samples", type=int, default=200,
                        help="number of samples to be generated")
    parser.add_argument("-np", "--node_prob", nargs="+", type=float,
                        help="the parameters range for node in diffusion model")
    parser.add_argument("-ep", "--edge_prob", nargs="+", type=float,
                        help="the parameters range for edges in diffusion model")
    parser.add_argument("-ss", "--seed_size", type=float, default=0.1,
                        help="The maximum percent of nodes been set as nodes")
    parser.add_argument("-cc", "--cpu_core", type=int, default=40)
    args = parser.parse_args()
    print("=== Setting Args ===")
    for arg, value in vars(args).items():
        print("{:>20} : {}".format(arg, value))

    data_path = os.path.join(args.data_path, "graph.pkl")
    save_path = os.path.join(args.data_path, "train_data")
    save_name = "{}_{}.pkl".format(args.diffusion, args.num_samples)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
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

    with open(data_path, "rb") as f:
        graph = pkl.load(f)
    graph = nx.DiGraph(nx.Graph(graph)) # Bi directed graph from original undirected graph
    node_rename = {n:i for i, n in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, node_rename)

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

    # random parameters
    assert args.edge_prob[0] <= args.edge_prob[1]
    if args.edge_prob[0] == args.edge_prob[1]:
        weights = np.ones((args.num_samples, E)) * args.edge_prob[0]
    else:
        weights = np.random.uniform(low=args.edge_prob[0], high=args.edge_prob[1], \
                                    size=(args.num_samples, E))
    
    assert args.node_prob[0] <= args.node_prob[1]
    if args.node_prob[0] == args.node_prob[1]:
        node_probs = np.ones((args.num_samples, N)) * args.node_prob[0]
    else:
        node_probs = np.random.uniform(low=args.node_prob[0], high=args.node_prob[1],
                                       size=(args.num_samples, N))

    # random seeds size
    if args.seed_size >=1:
        seeds_size = int(args.seed_size)
    else:
        seeds_size = np.random.randint(1, int(args.seed_size*N), 1)

    for s in range(args.num_samples):
        t0 = time.time()
        G = deepcopy(graph)

        # Setting parameters for edges nodes
        weight = weights[s]
        weight_map = {e:w for e, w in zip(G.edges, weight)}
        nx.set_edge_attributes(G, weight_map, "weight")

        node_prob = node_probs[s]
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
        simu_marginal = SIMU(args.diffusion, G, seeds, path="./src/utils/", cpu_core=args.cpu_core)
        dataset["simu_marginal"].append(simu_marginal)

        # DMP
        wadj = nx.adj_matrix(G).toarray()
        if args.diffusion == "SIR":
            dmp = DMP_SIR(wadj, node_prob)
        elif args.diffusion == "SIS":
            dmp = DMP_SIS(wadj, node_prob)
        dmp_simulation = dmp.run(list(seeds)).numpy()
        dataset["dmp_marginal"].append(dmp_simulation)
        
        d = aligning(simu_marginal, dmp_simulation)
        L1 = L1_error([simu_marginal], [d])

        print("{:^3} Sample Generated, Time={:.1f}s, Influence={:.1f}/{} DMP_L1={:.3f}".format(s, 
        time.time()-t0, simu_marginal[-1, :, -1].sum(), G.number_of_nodes(), L1))

# Saving
with open(os.path.join(save_path, save_name), "wb") as f:
    pkl.dump(dataset, f)






    
    



    

    

