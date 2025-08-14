from StructureDiscovery.JohnsonLindenstrauss.train import generate_data, train_model, evaluate_initial_distortion
from StructureDiscovery.JohnsonLindenstrauss.plot import plot_distortion, plot_variance

# Define constants
n = 100
d = 500
k = 30
BS = 20

# Set random seed
import numpy as np
np.random.seed(42)

# Generate data
X = generate_data(n, d)

# Train model
model, dists, s2s, init_dist = train_model(k, d, BS, X)

# Evaluate random initializations
mean_init_dist, min_init_dist = evaluate_initial_distortion(k, d, BS, X)

# Plot results
plot_distortion(dists, mean_init_dist, min_init_dist)
plot_variance(s2s)
