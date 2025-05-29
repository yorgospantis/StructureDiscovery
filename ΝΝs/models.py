import torch
import numpy as np
from scipy.stats import ortho_group

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def unit_Gaussian(n, d):
    return torch.normal(mean=0, std=1, size=(n, d), device=device, requires_grad=False)

def single_index(n, d, sigma=0.1, nonlin=None):
    X = unit_Gaussian(n, d)
    theta = torch.ones((d, 1), device=device)
    theta = theta / torch.norm(theta)
    if sigma == 0:
        return X, nonlin(X @ theta)
    return X, nonlin(X @ theta, i=100) + torch.randn(n, 1, device=device) * sigma

def multiple_index(n, d, theta, sigma=0, nonlin=None):
    X = unit_Gaussian(n, d)
    return X, (nonlin(X @ theta, i=100).mean(dim=1)).reshape(-1, 1) + torch.randn(n, 1, device=device) * sigma

def loss_fn(X, Y):
    return torch.mean((X - Y) ** 2)