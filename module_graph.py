from netgraph import Graph as plotGraph
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import module_ESN
def plot_graph_cool_esn(ESN, show_numbers = False, proxy_node = None ):
    G = outputgraph(ESN.W)
    cmap = 'RdYlGn'
    n = max(G.nodes())
    colors = { node:  'b'*(node == n)+  'gray'*(ESN.number_of_action_nodes<=node and node<n) + 'k'*(node <ESN.number_of_action_nodes )   for node in G.nodes() }
    if proxy_node :
        colors[proxy_node] = 'r'
    labs = {node: str(node) for node in G.nodes() if show_numbers}
    plt.figure(figsize=(10, 10))
    plotGraph(G, node_size=1,edge_width=0.5,node_edge_width = 0.1,arrows = True, edge_cmap=cmap, node_color=colors, node_labels = labs,node_label_offset=0.025, node_alpha=0.7)
    plt.show()

def outputgraph(adjacency_matrix):
    rows, cols = np.asarray(adjacency_matrix != 0).nonzero() 
    edges = [(b,a,  {'weight': adjacency_matrix[a,b]}) for a, b in zip(rows.tolist(), cols.tolist())]
    edges.sort()
    search_edges = [(a,b) for (a,b,c) in edges]
    for (a,b,c) in edges:
        if (b,a) in search_edges and a!=b:
            c['key'] = min(a,b)*1000 + max(a,b)
    gr = nx.DiGraph()
    gr.add_nodes_from(range(adjacency_matrix.shape[0]))
    gr.add_edges_from(edges)
    return gr



def plot_para(param_ESN, action_node, proxy_node):
    esn = module_ESN.EchoStateNetwork(param_ESN['n'],  spectral_radius=param_ESN['spectral_radius'], alpha = param_ESN['alpha'], 
                            avg_number_of_edges=param_ESN['avg_number_of_edges'], proxy_discard=param_ESN['proxy_discard'],
                            goal_discard=param_ESN['goal_discard'], measure_time=param_ESN['measure_time'], seed=param_ESN["seed"])
    print("action node:", action_node, " proxy node:", proxy_node, " goal node:", esn.n -1)
    action_values = np.zeros(len(esn.action_nodes))
    action_values[action_node] = 1
    states = esn.run( action_values)
    plt.plot(states[:,action_node],label = 'action node '+str(action_node))
    plt.plot(states[:,proxy_node],label = 'node '+str(proxy_node)+' (proxy)')
    plt.plot(states[:,-1],label = 'goal node')
    plt.legend()
    plt.show()
    plot_graph_cool_esn(esn, show_numbers=True, proxy_node=proxy_node)