# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 13:30:57 2018
@author: PLANZAGOM
"""
import random
import numpy as np
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
        
        AntColony.print_mat(self.phero_mat)
        
        print(self.build_best_path())
        
            
    def build_best_path(self):
        mat = AntColony.copy_mat(self.phero_mat)
        path = []
        j=0
        for i in range(self.N):
            nj = mat[j].index(max(mat[j]))
            mat[nj][j]=0
            path.append((j,nj))
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
            start = random.randint(0,self.N-1)
            (visited, tour_length, _) = self.ant_walk(start)
            visited_mat.append(visited)
            tour_length_list.append(tour_length)
            start_list.append(start)
        
        evap_mat = AntColony.create_mat(0,self.N,len(self.graph[0]))
        for i in range(len(visited_mat)):
            inv_tour_length = 1/tour_length_list[i]
            evap_mat = self.update_phero_mat(visited_mat[i], inv_tour_length, start_list[i], evap_mat)
        
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
        if i==j:
            return 0
        numerator = (self.phero_mat[i][j]**self.alpha)*((1/self.graph[i][j])**self.beta)
        denominator = 0
        for t in range(len(self.graph[i])):
            if t!=i:
                denominator += (self.phero_mat[i][t]**self.alpha)*((1/self.graph[i][t])**self.beta)
        prob = numerator/denominator
        return prob
        
    
if __name__=="__main__":
    
    graph = [[0,1,4,15],
             [1,0,8,4],
             [4,8,0,5],
             [15,4,5,0]]
    
    N = len(graph)
    M = len(graph[0])
    
    phero_mat = AntColony.create_mat(1, N, M)
    alpha = 1
    beta = 1
    rho = 0.5
    ac = AntColony(graph, phero_mat, alpha, beta, rho)
    num_ants = 100
    num_iters = 100
    ac.run(num_ants, num_iters)
