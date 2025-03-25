# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 22:08:15 2024

@author: dell
"""


import numpy as np
import networkx as nx
import multiprocessing as mp

def generate_33_symmetric_matrix(n):
  random_matrices = np.random.rand(n, 3, 3)
  symmetric_matrices = 0.5 * (random_matrices + np.transpose(random_matrices, (0, 2, 1)))
  return symmetric_matrices

num_steps = 500
num_nodes = 500
num_groups = 3

# Generate symmetric matrices
symmetric_matrices = generate_33_symmetric_matrix(num_steps)

# Transition matrix
transition_matrix = np.array([[0.8, 0.05, 0.15], [0.2, 0.75, 0.05], [0.25, 0.2, 0.55]])

# Initial states for each node
initial_states = np.random.choice([0, 1, 2], size=num_nodes)
def edge_density_dynsbm_fun(initial_states, num_steps, num_nodes, symmetric_matrices, transition_matrix):
  # Initialize edge density list
  edge_density_list = []

  # Iterate over steps
  for step in range(num_steps):
    if step == 0:
      current_states = initial_states.copy()
      previous_states = current_states.copy()

      # Compute beta_ql_t for all node pairs
      beta_ql_t = symmetric_matrices[step][previous_states[:, None], previous_states]

      # Generate edge time matrix
      edge_time = np.random.binomial(n=1, p=beta_ql_t)

      # Compute edge density
      edge_density_list.append(nx.density(nx.from_numpy_array(edge_time, create_using=nx.Graph)))
    else:
      previous_states = current_states.copy()

      # Compute beta_ql_t for all node pairs
      beta_ql_t = symmetric_matrices[step][previous_states[:, None], previous_states]

      # Generate edge time matrix
      edge_time = np.random.binomial(n=1, p=beta_ql_t)

      # Compute edge density
      edge_density_list.append(nx.density(nx.from_numpy_array(edge_time, create_using=nx.Graph)))

      #current_states = np.random.choice([0, 1, 2], size=num_nodes, p=transition_matrix[previous_states[:, None], :].T)

      for i in range(num_nodes):
        current_states[i] = np.random.choice([0, 1, 2], p=transition_matrix[previous_states[i]])
         
  return np.mean(np.array(edge_density_list))


def parallel_simulations(num_simulations, initial_states, num_steps, num_nodes, symmetric_matrices, transition_matrix):
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(edge_density_dynsbm_fun, [(initial_states, num_steps, num_nodes, symmetric_matrices, transition_matrix) for _ in range(num_simulations)])
    return results

mean_densities = parallel_simulations(3, initial_states, 500, 400, symmetric_matrices, transition_matrix)