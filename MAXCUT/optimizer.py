import torch
import numpy as np

def goemans_williamson(adj_matrix: np.ndarray, n: int) -> torch.Tensor:
    eigvals, eigvecs = np.linalg.eigh(adj_matrix)
    top_eigenvectors = eigvecs[:, np.argsort(eigvals)[-n:]]
    v = top_eigenvectors / np.linalg.norm(top_eigenvectors, axis=0)
    return torch.tensor(v, dtype=torch.float32)

def count_sign_mismatches(v: torch.Tensor, adj_matrix: torch.Tensor, r: torch.Tensor) -> int:
    S = (torch.matmul(v, r) > 0)
    cut_edges = torch.sum((adj_matrix > 0) & (S[:, None] != S[None, :])) // 2
    return cut_edges.item()

def compute_mismatch_weighted_derivatives(adj_matrix: torch.Tensor, v: torch.Tensor, m: torch.Tensor,
                                          sigma: torch.Tensor, num_samples: int):
    dL_dm = torch.zeros_like(m)
    dL_dsigma = torch.zeros_like(sigma)

    for _ in range(num_samples):
        sample_r = m + sigma * torch.randn_like(m)
        mismatches = count_sign_mismatches(v, adj_matrix, sample_r)
        dL_dm += mismatches * (sample_r - m) / (sigma ** 2)
        dL_dsigma += mismatches * ((sample_r - m) ** 2 / (sigma ** 3) - 1 / sigma)

    return dL_dm / num_samples, dL_dsigma / num_samples

def gradient_descent_step(adj_matrix: torch.Tensor, v: torch.Tensor, m: torch.Tensor, log_sigma: torch.Tensor,
                          lr_m: float, lr_sigma: float, num_samples: int, lambda_reg: float):
    sigma = torch.exp(log_sigma) + 0.5
    sigma.fill_(torch.max(sigma))
    dL_dm, dL_dsigma = compute_mismatch_weighted_derivatives(adj_matrix, v, m, sigma, num_samples)
    dL_dsigma -= 2 * lambda_reg * sigma

    m.data += lr_m * dL_dm
    log_sigma.data += lr_sigma * dL_dsigma
    log_sigma.data.clamp_(min=torch.log(torch.tensor(1e-3)), max=torch.log(torch.tensor(1.5)))

    return m, log_sigma

def optimize_max_cut(adj_matrix: torch.Tensor, num_vectors: int, num_steps: int,
                      lr_m: float, lr_sigma: float, lambda_reg: float, num_samples: int):
    v = goemans_williamson(adj_matrix.numpy(), num_vectors)
    m = torch.zeros(num_vectors, dtype=torch.float32, requires_grad=True)
    log_sigma = torch.zeros_like(m, requires_grad=True)
    sigmas = []
    max_sigmas = []
    cut_values = []

    for step in range(num_steps):
        cut_value = sum(count_sign_mismatches(v, adj_matrix, m + torch.exp(log_sigma) * torch.randn_like(m))
                        for _ in range(10)) / 10
        cut_values.append(cut_value)

        m, log_sigma = gradient_descent_step(adj_matrix, v, m, log_sigma, lr_m, lr_sigma, num_samples, lambda_reg)

        current_sigma = torch.exp(log_sigma).detach().clone()
        max_sigmas.append(torch.max(current_sigma).item())

        if step % 100 == 0:
            lr_m /= 1.02
            lr_sigma /= 1.03
            print(f"Step {step + 1}: Mean (m) = {m}, Std (sigma) = {current_sigma}, Cut value = {cut_value}")
            sigmas.append(current_sigma)

    return m, torch.exp(log_sigma), sigmas, cut_values, max_sigmas
