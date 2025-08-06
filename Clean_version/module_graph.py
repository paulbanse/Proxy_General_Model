from netgraph import Graph as plotGraph
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import module
def plot_graph_cool_esn(ESN, show_numbers = False):
    G = outputgraph(ESN.W)
    cmap = 'RdYlGn'
    n = max(G.nodes())
    colors = { node:  'g'*(node == n)+  'k'*(16<=node and node<n) + 'r'*(node <16)   for node in G.nodes() }
    labs = {node: str(node) for node in G.nodes() if show_numbers}
    plt.figure(figsize=(10, 10))
    plotGraph(G, node_size=1,edge_width=0.5,node_edge_width = 0.1,arrows = True, edge_cmap=cmap, node_color=colors, node_labels = labs,node_label_offset=0.025)

def outputgraph(adjacency_matrix):
    rows, cols = np.asarray(adjacency_matrix != 0).nonzero() 
    edges = [(b,a,  {'weight': adjacency_matrix[a,b]}) for a, b in zip(rows.tolist(), cols.tolist())]
    #
    gr = nx.DiGraph()
    gr.add_edges_from(edges)
    return gr
