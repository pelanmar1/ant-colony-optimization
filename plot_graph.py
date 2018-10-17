# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 21:08:38 2018

@author: PLANZAGOM
"""

import networkx as nx
import matplotlib.pyplot as plt


def pause():
    input("Press Enter to continue ...")


def matrix_to_edges(mat):
    edges = []
    for i in range(len(mat)):
        for j in range(i+1, len(mat[0])):
            edges.append((i, j))
    return edges


def get_weights_freq(mat):
    costs = []
    edges = matrix_to_edges(mat)
    for i, j in edges:
            cost = mat[i][j]
            costs.append(cost)
    max_cost = max(costs)
    weights = [c/max_cost for c in costs]
    return weights

def get_costs(mat):
    # {(0, 1): 1.0, (0, 2): 4, (0, 3): 4, (1, 2): 4, (1, 3): 4, (2, 3): 1.0}
    costs = {}
    edges = matrix_to_edges(mat)
    for (i, j) in edges:
        costs[(i, j)] = mat[i][j]
    return costs

def get_weights_order(mat):
    costs = []
    edges = matrix_to_edges(mat)
    for i, j in edges:
        cost = mat[i][j]
        costs.append(cost)
    weights = [i+1 for i in sorted(range(len(costs)), key=costs.__getitem__, reverse=True)]
    return weights

def get_weights(mat, best_path):
    weights = []
    edges = matrix_to_edges(mat)
    for i, j in edges:
        if (i,j) in best_path or (j, i) in best_path:
            weights.append(2)
        else:
            weights.append(0.5)
    return weights

def draw_graph(mat, best_path, cost_mat):
    K = 2
    weights = get_weights(mat, best_path)
    edges = matrix_to_edges(mat)
    fig = plt.figure()
    G = nx.Graph()
    node_names = [i for i in range(len(mat))]
    node_labels = {}
    for i in node_names:
        node_labels[i] = r'$'+str(i)+'$'
    G.add_nodes_from(node_names)
    z = 0
    for i, j in edges:
        G.add_edge(i, j, weight=weights[z]*K)
        z += 1
        
    pos = nx.circular_layout(G)
    eds = G.edges()
    wes = [G[u][v]['weight'] for u, v in eds]

    labels = get_costs(cost_mat)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    nx.draw(G, pos, edges=eds, width=wes, node_color="cyan", with_labels = True)

    return fig


# graph = [[0,1,4,15],
#          [1,0,8,4],
#          [4,8,0,5],
#          [15,4,5,0]]

# graph = [[0.0,  0.96,  0.92,  0.67],
#         [0.96,  0.0,  0.67,  0.92],
#         [0.92,  0.67,  0.0,  0.96],
#         [0.67,  0.92,  0.96,  0.0 ]]
#
# draw_graph(graph)

