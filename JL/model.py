import torch
import torch.nn as nn
import numpy as np

class DistortionOptimizer(nn.Module):
    def __init__(self, k, d, batch_size):
        super(DistortionOptimizer, self).__init__()
        self.k = k
        self.d = d
        self.kd = k * d
        self.batch_size = batch_size

        self.mean = nn.Parameter(torch.zeros(self.kd))
        self.log_variance = nn.Parameter(torch.zeros(1))

    def forward(self):
        std_dev = torch.exp(0.5 * self.log_variance)
        epsilon = torch.randn(self.batch_size, self.kd)
        A_flat = self.mean + std_dev * epsilon
        return A_flat.view(self.batch_size, self.k, self.d)

    def compute_distortion(self, A, X):
        AX = torch.matmul(A, X.T)
        distortions = torch.norm(1 / np.sqrt(self.k) * AX, dim=1) ** 2 - 1
        return torch.max(torch.abs(distortions))

    def gradient(self, A, X):
        return self.compute_distortion(A, X)

