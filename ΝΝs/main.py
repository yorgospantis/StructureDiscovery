from train import train
from activations import relui, tanh
from plot_weights import plot_weight_trajectory

if __name__ == "__main__":
    result = train(n=50, d=2, steps=10_000, h=1000,
                   lda=1e-5, lr=0.8, lr_b=0.5,
                   nonlin=tanh, act=relui)
    plot_weight_trajectory(result)
