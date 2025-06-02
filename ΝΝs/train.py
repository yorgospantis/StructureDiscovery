import torch
import numpy as np
from tqdm import tqdm
from StructureDiscovery.NNs.model import unit_Gaussian, loss_fn, single_index, multiple_index
from activations import relui

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(n, d, h, theta=None, nonlin=None, act=None, teacher_model=single_index,
          lr=0.01, lr_b=1e-3, steps=100_000, lda=1e-7, sigma=0, points=5000,
          train_fin=False, random_seed=0, decay_stepsize=False):

    torch.manual_seed(random_seed)
    step_list = np.exp(np.linspace(0, np.log(steps), points)).astype(int)
    step_list = np.unique(step_list) - 1

    result = np.zeros([len(step_list), h, d])
    W0 = torch.randn(d, h, device=device) * (1 / d) ** 0.5
    a = torch.randn(h, 1, device=device) * (1 / h)
    b = torch.randn(1, h, device=device) * (1 / h)

    eta1 = (lr * h ** 0.5)
    etab = (lr_b ** 2 * h ** 0.5) ** 2

    W = W0.clone(); W.requires_grad = True
    b.requires_grad = True
    if train_fin:
        a.requires_grad = True

    def forward(X):
        return act(X @ W + b) @ a

    count = 0
    for i in tqdm(range(steps)):
        eta_i = eta1 / (i + 1) if decay_stepsize else eta1

        if teacher_model == multiple_index:
            X, Y = teacher_model(n, d, theta, sigma, nonlin)
        else:
            X, Y = teacher_model(n, d, sigma, nonlin)

        loss = loss_fn(Y, forward(X))
        loss.backward()

        with torch.no_grad():
            W -= eta_i * (W.grad + lda * W)
            b -= etab * b.grad
            if train_fin:
                a -= eta_i * (a.grad + lda * a)

        W.grad.zero_(); b.grad.zero_()
        if train_fin: a.grad.zero_()

        if count < len(step_list) and i == step_list[count]:
            for j in range(d):
                result[count, :, j] = W[j, :].detach().cpu().numpy()
            count += 1

        if i % 1000 == 0:
            print(f"Training loss: {loss.item()}")

    test_n = 10000
    if teacher_model == multiple_index:
        theta = torch.Tensor(ortho_group.rvs(dim=d)[:, :1]).to(device)
        test_X, test_Y = teacher_model(test_n, d, theta, sigma, nonlin)
    else:
        test_X, test_Y = teacher_model(test_n, d, sigma, nonlin)

    test_Y_pred = forward(test_X)
    loss_test = loss_fn(test_Y, test_Y_pred)
    print(f"Test loss: {loss_test.item()}")

    return result
