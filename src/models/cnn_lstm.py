"""
Flagship CNN-LSTM model for Remaining Useful Life prediction.
Combines 1D temporal convolutions (extracting multi-sensor local dynamics)
with an LSTM layer (capturing long-term degradation trajectories).
"""
import torch
import torch.nn as nn


class CNNLSTMRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        conv_channels: list = [32, 64],
        kernel_size: int = 3,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # 1D Convolutional feature extractors across temporal sequence
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=conv_channels[0],
            kernel_size=kernel_size,
            padding="same",
        )
        self.bn1 = nn.BatchNorm1d(conv_channels[0])
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(
            in_channels=conv_channels[0],
            out_channels=conv_channels[1],
            kernel_size=kernel_size,
            padding="same",
        )
        self.bn2 = nn.BatchNorm1d(conv_channels[1])
        self.relu2 = nn.ReLU()
        
        # Recurrent layer capturing long-term degradation trend
        self.lstm = nn.LSTM(
            input_size=conv_channels[1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )
        
        # Regression head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, n_features)
        # Conv1d expects (batch, n_features, seq_len)
        x_conv = x.permute(0, 2, 1)
        x_conv = self.relu1(self.bn1(self.conv1(x_conv)))
        x_conv = self.relu2(self.bn2(self.conv2(x_conv)))
        
        # Reshape for LSTM: (batch, seq_len, conv_channels[1])
        x_lstm_in = x_conv.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_lstm_in)
        
        # Take the representation at the final sequence step
        last_hidden = lstm_out[:, -1, :]
        out = self.head(last_hidden)
        return out
