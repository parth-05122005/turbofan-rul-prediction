"""
Flower NumPyClient implementation for Federated Turbofan Predictive Maintenance.
Trains a local CNN-LSTM model on private factory shard data.
"""
from typing import Dict, List, Tuple
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl

from src.models.cnn_lstm import CNNLSTMRegressor
from src.eval.metrics import compute_all_metrics


class TurbofanClient(fl.client.NumPyClient):
    """
    Simulated factory client participating in FedAvg training.
    Maintains a private shard of engine sensor trajectories.
    """
    def __init__(
        self,
        cid: str,
        model: CNNLSTMRegressor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cpu",
        local_epochs: int = 2,
        lr: float = 1e-3,
    ):
        self.cid = cid
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.criterion = nn.HuberLoss(delta=1.0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """Extracts model state dict as list of numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Loads parameter arrays into PyTorch model state dict."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        """Trains model on local factory data."""
        self.set_parameters(parameters)
        self.model.train()
        
        epochs = config.get("local_epochs", self.local_epochs) if config else self.local_epochs
        total_loss = 0.0
        n_samples = 0
        
        for epoch in range(int(epochs)):
            for X_b, y_b in self.train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(X_b)
                loss = self.criterion(pred, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item() * len(y_b)
                n_samples += len(y_b)
                
        avg_loss = total_loss / max(n_samples, 1)
        num_examples = len(self.train_loader.dataset)
        return self.get_parameters(config), num_examples, {"train_loss": avg_loss}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict[str, float]]:
        """Evaluates model on local validation split."""
        self.set_parameters(parameters)
        self.model.eval()
        
        val_loss = 0.0
        val_samples = 0
        preds, trues = [], []
        
        with torch.no_grad():
            for X_b, y_b in self.val_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                pred = self.model(X_b)
                loss = self.criterion(pred, y_b)
                val_loss += loss.item() * len(y_b)
                val_samples += len(y_b)
                preds.extend(pred.cpu().numpy().ravel())
                trues.extend(y_b.cpu().numpy().ravel())
                
        avg_loss = val_loss / max(val_samples, 1)
        metrics = compute_all_metrics(np.array(trues), np.array(preds))
        num_examples = len(self.val_loader.dataset)
        return float(avg_loss), num_examples, {
            "mae": metrics["MAE"],
            "rmse": metrics["RMSE"],
            "nasa_score": metrics["NASA_Score"],
        }
