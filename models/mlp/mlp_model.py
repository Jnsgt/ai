import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

        with torch.no_grad():
            for name, param in self.named_parameters():
                if "weight" in name:
                    nn.init.constant_(param, 0.1)
                elif "bias" in name:
                    nn.init.constant_(param, 0.2)

    def forward(self, x):
        return self.net(x)