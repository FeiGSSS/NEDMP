import networkx as nx
import pickle as pkl

with open("raw_data/soc-dolphins.mtx") as f:
    edges = []
    for line in f:
        if "%" in line:
            continue
        line = line.strip().split(" ")
        e = [int(x) for x in line]
        if len(e) > 2:
            continue
        edges.append(e)

G = nx.Graph()
G.add_edges_from(edges)
G = nx.relabel_nodes(G, {n:cnt for cnt, n in enumerate(G.nodes)})

with open("graph.pkl", "wb") as f:
        pkl.dump(G, f)
