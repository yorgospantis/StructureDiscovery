import torch
import numpy as np

def relui(X, i=2, normalize=False):
    relu_i = 0.5 * torch.log(1 + torch.exp(2 * X))
    if normalize:
        return np.sqrt(2 * np.pi / (np.pi - 1)) * (relu_i - 1 / np.sqrt(2 * np.pi))
    return relu_i

def tanh(X, normalize=False):
    X = torch.tanh(X)
    if normalize:
        c = 0.627928
        return X / c
    return X

def quadratic(X, normalize=True):
    if normalize:
        return (X ** 2 - 1) / np.sqrt(2)
    return X ** 2 / 2

def mypoly(X):
    return X ** 3 - X

def linear(X):
    return X
