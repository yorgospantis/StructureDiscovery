import torch
import numpy as np
import torch.optim as optim
from distortion_model import DistortionOptimizer

def generate_data(n, d):
    return torch.nn.functional.normalize(torch.rand((n, d)), dim=-1)

def train_model(k, d, BS, X, steps=5000, lr=0.01):
    model = DistortionOptimizer(k, d, BS)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dists, s2s = [], []
    initial_distortion = None

    for step in range(steps):
        optimizer.zero_grad()
        A = model()
        max_distortion = model.compute_distortion(A, X)

        if step == 0:
            initial_distortion = max_distortion.item()

        loss = model.gradient(A, X)
        loss.backward()
        optimizer.step()

        dists.append(max_distortion.item())
        s2s.append(torch.exp(0.5 * model.log_variance).item())

        if step % 1000 == 0:
            print(f"Step {step}, Max Distortion: {max_distortion.item():.4f}, "
                  f"Sigma: {torch.exp(0.5 * model.log_variance).item():.4f}")

        if max_distortion.item() < 1e-2:
            break

    return model, dists, s2s, initial_distortion

def evaluate_initial_distortion(k, d, BS, X, samples=1000):
    distortions = []
    for _ in range(samples):
        model = DistortionOptimizer(k, d, BS)
        A = model()
        distortions.append(model.compute_distortion(A, X).detach().item())
    return np.mean(distortions), np.min(distortions)
