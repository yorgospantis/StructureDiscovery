from StructureDiscovery.NeuralNetworks.train import train
from StructureDiscovery.NeuralNetworks.activations import relui, tanh
from StructureDiscovery.NeuralNetworks.plot import plot_weight_trajectory

result = train(n=50, d=2, steps=10_000, h=1000, lda=1e-5, lr=0.8, lr_b=0.5, nonlin=tanh, act=relui)
plot_weight_trajectory(result)