"""
Shared training and evaluation loop utilities for PyTorch deep learning sequence models.
"""
from typing import Dict, Tuple, Optional
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.windowing import TurbofanDataset
from src.eval.metrics import compute_all_metrics


def train_deep_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Standard training loop with Huber loss, Adam optimizer, LR scheduling, and early stopping.
    """
    model.to(device)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": []}
    best_loss = float("inf")
    best_weights = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        n_samples = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * len(y_b)
            n_samples += len(y_b)
        train_loss = running_loss / n_samples

        # Validation phase
        model.eval()
        val_preds, val_trues = [], []
        val_running_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred = model(X_b)
                loss = criterion(pred, y_b)
                val_running_loss += loss.item() * len(y_b)
                val_samples += len(y_b)
                val_preds.extend(pred.cpu().numpy().ravel())
                val_trues.extend(y_b.cpu().numpy().ravel())
                
        val_loss = val_running_loss / val_samples
        val_metrics = compute_all_metrics(np.array(val_trues), np.array(val_preds))
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_metrics["MAE"])
        history["val_rmse"].append(val_metrics["RMSE"])

        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val MAE: {val_metrics['MAE']:.2f} | "
                f"Val RMSE: {val_metrics['RMSE']:.2f}"
            )

        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    return model, history


def evaluate_model_on_windows(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    device: str = "cpu",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Runs model inference on given windows and returns predictions and metrics.
    """
    model.eval()
    model.to(device)
    dataset = TurbofanDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    preds = []
    with torch.no_grad():
        for X_b, _ in loader:
            X_b = X_b.to(device)
            out = model(X_b)
            preds.extend(out.cpu().numpy().ravel())
            
    preds_arr = np.array(preds)
    metrics = compute_all_metrics(y, preds_arr)
    return preds_arr, metrics
