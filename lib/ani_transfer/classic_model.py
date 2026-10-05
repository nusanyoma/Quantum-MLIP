import torch
import torch.nn as nn


class ClassicNet(nn.Module):
    def __init__(self, nwires, device):
        super().__init__()
        self.c_net = nn.Linear(nwires, nwires).to(device)
        self.activation = nn.CELU(0.1).to(device)

    def forward(self, inputs):
        c_out = self.c_net(inputs)
        c_out = self.activation(c_out)
        return c_out
