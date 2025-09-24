from StructureDiscovery.NeuralNetworks.train import train
from StructureDiscovery.NeuralNetworks.activations import relui, tanh
from StructureDiscovery.NeuralNetworks.plot import plot_weight_trajectory

result = train(n=50, d=2, steps=20_000, h=1000, lda=1e-4, lr=1, lr_b=1, epsilon=1e-3)