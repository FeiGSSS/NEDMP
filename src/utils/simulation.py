# -*- encoding: utf-8 -*-
'''
@File    :   simulation.py
@Time    :   2021/06/15 10:07:13
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import os
import subprocess
import numpy as np

def SIMU(diff, graph, seeds, path, num_simulation=10000, cpu_core=40):
    """C implementation of Simulation

    Args:
        diff (str]): "SIR" or "SIS"
        graph (nx.DiGraph): the graph structure
        seeds (list): list of seeds
        num_simulation (int): [description]
        cpu_core (int): [description]

    Raises:
        RuntimeError: [description]

    Returns:
        [type]: [description]
    """
    # write txt file for simulation
    txt = open(".tmp_simulation.txt", "w")
    N = graph.number_of_nodes()
    E = graph.number_of_edges()
    S = len(seeds)
    txt.write("{} {} {}\n\n\n".format(N, E, S))
    seeds = [str(s) for s in seeds]
    txt.write(" ".join(seeds)+"\n\n\n")

    node_prob = {n:graph.nodes[n]["node_prob"] for n in graph.nodes()}
    for n, p in node_prob.items():
        txt.write("{} {}\n".format(n, p))
    txt.write("\n\n")

    edge_prob = {(i, j):graph[i][j]["weight"] for (i, j) in graph.edges()}
    for e, p in edge_prob.items():
        txt.write("{} {} {}\n".format(e[0], e[1], p))
    txt.close()
    
    # run simulation
    cmd = path+"/simulation -dataset .tmp_simulation.txt \
            -model {} -mc {} -maxSimSteps 50 -cpuCores {}".\
            format(diff, num_simulation, cpu_core)
    res = subprocess.getstatusoutput(cmd)
    os.remove(".tmp_simulation.txt")
    if int(res[0]) != 0:
        print(res[1])
        raise RuntimeError
    
    # read simulation results
    with open("spreadProcess.txt", "r") as f:
        spread = f.readlines()
    T, N, K = [int(x) for x in spread[0].strip().split(" ")]
    spreading = np.zeros((T*N, K))
    cnt = 0
    for line in spread[1:]:
        if len(line.strip()) == 0:
            continue
        for k, x in enumerate(line.strip().split(" ")):
            spreading[cnt, k] = float(x)
        cnt += 1
    assert cnt == T*N
    spreading = spreading.reshape(T, N, K)
    os.remove("spreadProcess.txt")
    return spreading


# import networkx as nx
# import ndlib.models.ModelConfig as mc
# import ndlib.models.epidemics as ep
# from ndlib.utils import multi_runs
# import copy
# import collections


# def SIR_ndlib(graph, beta, gamma, seeds, num_iterations):
#     # Model selection
#     model = ep.SIRModel(graph)
#     config = mc.Configuration()
#     config.add_model_parameter('beta', beta)
#     config.add_model_parameter('gamma', gamma)
#     config.add_model_initial_configuration("Infected", seeds)
#     model.set_initial_status(config)
#     iterations = model.iteration_bunch(num_iterations, node_status=True)

#     status_list = []
#     I, R = [], []
#     for k, v in iterations[0]["status"].items():
#         if v==1:
#             I.append(k)
#         elif v==2:
#             R.append(k)
#     status = [I, R]
#     status_list.append(status)

#     # t>=1
#     for ite in iterations[1:]:
#         status_old = status_list[-1]
#         I = copy.deepcopy(status_old[0])
#         R = copy.deepcopy(status_old[1])
        
#         status_updata = ite["status"]
#         for k, v in status_updata.items():
#             if v==1:
#                 I.append(k)
#             elif v==2:
#                 I.remove(k)
#                 R.append(k)
#         status_new = [I, R]
#         status_list.append(status_new)

#     return status_list


