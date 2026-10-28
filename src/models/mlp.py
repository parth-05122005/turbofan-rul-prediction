"""
Multi-Layer Perceptron (MLP) baseline for sequence sliding windows.
Flattens (batch, seq_len, n_features) into 2D vector, ignoring temporal order.
"""
import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden_dims: list = [128, 64], dropout: float = 0.2):
        super().__init__()
        in_dim = seq_len * n_features
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        x_flat = x.view(x.size(0), -1)
        return self.net(x_flat)
