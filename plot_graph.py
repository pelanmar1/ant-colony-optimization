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
        for j in range(i+1,len(mat[0])):
            edges.append((i, j))
    return edges

def get_weights(mat):
    costs = []
    edges = matrix_to_edges(mat)
    for i,j in edges:
            cost = mat[i][j]
            costs.append(cost)
    max_cost = max(costs)
    weights = [c/max_cost for c in costs]
    return weights

def draw_graph(mat):
    K = 3
    weights = get_weights(mat)
    fig=plt.figure()
    G = nx.Graph()
    
    G.add_nodes_from([i for i in range(len(mat))])
    z = 0
    for i,j in edges:
        G.add_edge(i,j, weight=weights[z]*K)
        z += 1
        
    pos = nx.circular_layout(G)
    eds = G.edges()
    wes = [G[u][v]['weight'] for u,v in eds]
    
    nx.draw(G, pos, edges=eds, width=wes)
    


graph = [[0,1,4,15],
         [1,0,8,4],
         [4,8,0,5],
         [15,4,5,0]]

graph = [[0.0,  0.96,  0.92,  0.67],  
        [0.96,  0.0,  0.67,  0.92],  
        [0.92,  0.67,  0.0,  0.96],  
        [0.67,  0.92,  0.96,  0.0 ]]

draw_graph(graph)

