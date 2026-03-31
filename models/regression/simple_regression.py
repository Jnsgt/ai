import torch
import torch.nn as nn


class SimpleRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

        with torch.no_grad():
            self.linear.weight[:] = torch.tensor([[2.0]])
            self.linear.bias[:] = torch.tensor([1.0])

    def forward(self, x):
        return self.linear(x)