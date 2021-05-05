import torch
import numpy as np
import torch.nn as nn

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
    nn.Linear(D_in, H),
    nn.ReLU(),
    nn.Linear(H, D_out),
)

loss_fn = nn.MSELoss(reduction='sum')

lr = 1e-4

for t in range(1000):
    # forward pass
    y_pred = model(x)

    # compute loss
    loss = loss_fn(y_pred, y)
    print(t, loss)

    model.zero_grad()

    # backward pass,compute the gradient
    loss.backward()
    # update weights of w1 and w2
    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad
