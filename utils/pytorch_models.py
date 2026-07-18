# DL не должен тащит за собой отдельный запуск.
# PyTorch-модель совместима с текущим интерфейсом и запускается
# тем же python main.py.

# MLP — Multilayer Perceptron.

import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin

def resolve_device(device_name):
    if device_name == "auto":
        if torch.cuda.is_avalilable():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)
    
    print(f"Using device: {device}.")

    return device

def get_activation(name):
    if name == "relu":
        return nn.Relu()

    if name == "tanh":
        return nn.Tanh()

    if name == "sigmoid":
        return nn.Sigmoid()

    else:
        raise ValueError(f"Unknown activation: {name}.")

class TitanicMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation_name):
        super().__init__()

        activation_layer = get_activation(activation_name)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_layer,
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


