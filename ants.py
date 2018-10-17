# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 13:30:57 2018
@author: PLANZAGOM
"""
import random
import numpy as np
import plot_graph
import matplotlib.pyplot as plt
class AntColony:
    
    def __init__(self, graph, phero_mat, alpha, beta, rho):
        self.graph = graph
        self.phero_mat = phero_mat
        self.alpha = alpha
        self.beta = beta
        self.N = len(graph)
        self.rho = rho
    
    def run(self, num_ants, num_iters):
        for i in range(num_iters):
            self.start_ants(num_ants)
        print("Final Pheromone Matrix:")
        AntColony.print_mat(self.phero_mat)
        print("Strongest edges: ")
        print(self.build_best_path())

    def build_best_path(self):
        mat = AntColony.copy_mat(self.phero_mat)
        path = []
        j = 0
        for i in range(self.N):
            nj = mat[j].index(max(mat[j]))
            mat[nj][j] = 0
            path.append((j, nj))
            j = nj
            
        return path
    
    def copy_mat(mat):
        n_mat = []
        for i in range(len(mat)):
            row = []
            for j in range(len(mat[0])):
                row.append(mat[i][j])
            n_mat.append(row)
        return n_mat
    
    def create_mat(val, n, m, diag=True):
        mat = []
        for i in range(n):
            row = []
            for j in range(m):
                if diag and i==j:
                    row.append(0)
                else:
                    row.append(val)
            mat.append(row)
        return mat

    def print_mat(mat, r=2):
        for i in range(len(mat)):
            row = ""
            for j in range(len(mat[0])):
                row += str(float(round(mat[i][j], r))) + "  "
            print(row)
        print("")
    
    def start_ants(self, num_ants):
        visited_mat = []
        tour_length_list = []
        start_list = []
        
        for ant in range(num_ants):
            start = random.randint(0, self.N-1)
            (visited, tour_length, _) = self.ant_walk(start)
            visited_mat.append(visited)
            tour_length_list.append(tour_length)
            start_list.append(start)

        evap_mat = AntColony.create_mat(0,self.N,len(self.graph[0]))
        for i in range(len(visited_mat)):
            inv_tour_length = 1/tour_length_list[i]
            self.draw_network(i)
            evap_mat = self.update_phero_mat(visited_mat[i], inv_tour_length, start_list[i], evap_mat)


    def draw_network(self, file_name):
        fig = plot_graph.draw_graph(self.phero_mat, self.build_best_path(), self.graph)
        # plot_graph.pause()
        fig.savefig("img/" + str(file_name) + ".png")
        plt.close(fig)

    def ant_walk(self, start):
        i = start
        visited = [start]
        tour_length = 0
        while len(visited) < self.N:
            path_probs = []
            paths = self.graph[i]
            for j in range(len(paths)):
                path_probs.append(self.calc_prob(i,j))
            choice = self.make_choice(path_probs, visited)
            if choice is None:
                print("ERROR")
            visited.append(choice)
            tour_length += graph[i][choice]
            i = choice
        # Return to start
        visited.append(start)
        tour_length += graph[choice][start]
        return (visited, tour_length, start)
    
    def make_choice(self, path_probs, visited):
        eps = 0.001
        not_allowed_accum = 0
        for i in visited:
            not_allowed_accum += path_probs[i]
            path_probs[i] = 0.0
        denominator = len([1 for i in path_probs if i > 0])
        add_prob = not_allowed_accum / denominator
        for j in range(len(path_probs)):
            if j not in visited and path_probs[j]>0:
                path_probs[j] += add_prob
        if abs(sum(path_probs)-1)>eps:
            return None
        choice = np.random.choice(len(path_probs), p=path_probs)
        return choice
        
    def update_phero_mat(self, visited, pherm_sum, start, evap_mat):
        for i in range(len(visited)-1):
            a = visited[i]
            b = visited[i+1]
            if evap_mat[a][b] == 0:
                new_pheromone = self.phero_mat[a][b]*self.rho + pherm_sum
            else:
                new_pheromone = self.phero_mat[a][b] + pherm_sum
            self.phero_mat[a][b] = new_pheromone
            self.phero_mat[b][a] = new_pheromone
            evap_mat[a][b] = 1
            evap_mat[b][a] = 1
        return evap_mat

    def calc_prob(self, i, j):
        if i == j:
            return 0
        numerator = (self.phero_mat[i][j]**self.alpha)*((1/self.graph[i][j])**self.beta)
        denominator = 0
        for t in range(len(self.graph[i])):
            if t != i:
                denominator += (self.phero_mat[i][t]**self.alpha)*((1/self.graph[i][t])**self.beta)
        prob = numerator/denominator
        return prob
        
    
if __name__=="__main__":
    
    # graph = [[0,1,4,15],
    #          [1,0,8,4],
    #          [4,8,0,5],
    #          [15,4,5,0]]
    # Cities
    city_names = ["New York", "Los Angeles", "Chicago", "Minneapolis", "Denver", "Dallas", "Seattle",
                  "Boston", "San Francisco", "St. Louis", "Houston", "Phoenix", "Salt Lake City"]
    # Distance matrix
    graph = [
        [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],  # New York
        [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],  # Los Angeles
        [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],  # Chicago
        [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],  # Minneapolis
        [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],  # Denver
        [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],  # Dallas
        [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],  # Seattle
        [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],  # Boston
        [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],  # San Francisco
        [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],  # St. Louis
        [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],  # Houston
        [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],  # Phoenix
        [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0]]  # Salt Lake City

    N = len(graph)
    M = len(graph[0])
    
    phero_mat = AntColony.create_mat(1, N, M)
    alpha = 1
    beta = 1
    rho = 0.5
    ac = AntColony(graph, phero_mat, alpha, beta, rho)
    num_ants = 30
    num_iters = 1
    ac.run(num_ants, num_iters)
    
