import torch
import numpy as np
from maxcut_utils import generate_adjacency_matrix, max_cut
from maxcut_optimizer import optimize_max_cut
from maxcut_plotting import plot_sigma_evolution, plot_cut_values

# Seeds
torch.manual_seed(42)
np.random.seed(42)

# Generate graph
graph_size = 15
prob_edge = 0.6
adj_matrix = generate_adjacency_matrix(graph_size, prob_edge)
print("Generated Adjacency Matrix:")
print(adj_matrix)

# Compute exact MaxCut
max_cut_value, best_partition = max_cut(adj_matrix)
print("Max Cut Value:", max_cut_value)
print("Best Partition:", best_partition)

# Set random seed for optimization
torch.manual_seed(43)
np.random.seed(43)

# Optimization params
num_vectors = adj_matrix.shape[0]
num_steps = 5000
lr_m = 0.01
lr_sigma = 0.001
lambda_reg = 1
num_samples = 100

# Run optimization
m, sigma, sigmas, cut_values, max_sigmas = optimize_max_cut(
    adj_matrix, num_vectors, num_steps, lr_m, lr_sigma, lambda_reg, num_samples
)

# Plot results
plot_sigma_evolution(max_sigmas)
plot_cut_values(cut_values, max_cut_value)