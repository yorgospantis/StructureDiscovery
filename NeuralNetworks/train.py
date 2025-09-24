import torch
import numpy as np
from tqdm import tqdm
from StructureDiscovery.NeuralNetworks.model import unit_Gaussian, loss_fn, single_index, multiple_index
from StructureDiscovery.NeuralNetworks.activations import relui

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(n, d, h, theta=None, nonlin=tanh, act=relui, teacher_model=single_index,
                     lr=0.01, lr_b=1e-3, steps=10_000, lda=1e-5, sigma=0,
                     points=5_000, train_fin=False, random_seed=0,
                     epsilon=1e-6, perturb_radius=0.005):

    torch.manual_seed(random_seed)

    step_list = np.exp(np.arange(0, np.log(steps), np.log(steps)/points)).astype(int)
    step_list = np.unique(step_list) - 1
    result = np.zeros([len(step_list), h, d])

    # Initialize weights
    W = torch.normal(0, (1/d)**0.5, size=(d, h), device=device, requires_grad=True)
    a = torch.normal(0, (1/h)**1, size=(h, 1), device=device)
    b = torch.normal(0, (1/h), size=(1, h), device=device, requires_grad=True)
    if train_fin:
        a.requires_grad = True

    def forward(X):
        return act(X @ W + b) @ a

    count = 0
    for i in tqdm(range(steps)):
        # Generate training data
        if teacher_model == multiple_index:
            X, Y = teacher_model(n, d, theta, sigma, nonlin)
        else:
            X, Y = teacher_model(n, d, sigma, nonlin)

        # Compute loss
        loss_train = loss_fn(Y, forward(X))
        loss_train.backward()

        with torch.no_grad():
            # Compute gradient norm for conditional perturbation
            grad_norm = torch.norm(W.grad)

            # Standard gradient descent + weight decay
            W -= lr * (W.grad + lda * W)
            b -= lr_b * b.grad
            if train_fin:
                a -= lr * (a.grad + lda * a)

            # Conditional perturbation if gradient is small
            if grad_norm < epsilon:
                # Perturb W and b uniformly within a ball
                W += perturb_radius * torch.randn_like(W)
                b += perturb_radius * torch.randn_like(b)
                if train_fin:
                    a += perturb_radius * torch.randn_like(a)

        # Reset gradients
        W.grad.zero_()
        b.grad.zero_()
        if train_fin:
            a.grad.zero_()

        # Store weights at step_list
        if count < len(step_list) and i == step_list[count]:
            for j in range(d):
                result[count, :, j] = W[j, :].detach().cpu().numpy()
            count += 1

        if i % 1000 == 0:
            print(f'Training loss: {loss_train.item()} | grad_norm: {grad_norm.item()}')

    # Evaluate on test set
    test_n = 10000
    if teacher_model == multiple_index:
        theta = torch.Tensor(ortho_group.rvs(dim=d)[:, :1]).to(device)
        test_X, test_Y = teacher_model(test_n, d, theta, sigma, nonlin)
    else:
        test_X, test_Y = teacher_model(test_n, d, sigma, nonlin)

    test_Y_pred = forward(test_X)
    loss_test = loss_fn(test_Y, test_Y_pred)
    print(f'Test loss: {loss_test.item()}')

    return result
