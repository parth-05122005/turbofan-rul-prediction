"""
Flower Server strategy configuration for FedAvg federated aggregation.
"""
from typing import Callable, Dict, List, Optional, Tuple
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import Metrics, Scalar

from src.models.cnn_lstm import CNNLSTMRegressor
from src.eval.metrics import compute_all_metrics


def weighted_average_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates metrics from multiple clients weighted by number of examples."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    aggregated = {}
    
    if total_examples == 0:
        return aggregated

    for _, m in metrics:
        for k, v in m.items():
            if isinstance(v, (int, float)):
                aggregated[k] = aggregated.get(k, 0.0) + v * _
                
    for k in aggregated:
        aggregated[k] = aggregated[k] / total_examples
        
    return aggregated


def create_centralized_eval_fn(
    model_template: CNNLSTMRegressor,
    test_loader: DataLoader,
    device: str = "cpu",
) -> Callable[[int, List[np.ndarray], Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]:
    """
    Returns an evaluation function that scores the global aggregated model
    on the centralized held-out test/validation set after each communication round.
    """
    criterion = nn.HuberLoss(delta=1.0)
    
    def evaluate(
        server_round: int, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        params_dict = zip(model_template.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model_template.load_state_dict(state_dict, strict=True)
        model_template.to(device)
        model_template.eval()
        
        running_loss = 0.0
        n_samples = 0
        preds, trues = [], []
        
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred = model_template(X_b)
                loss = criterion(pred, y_b)
                running_loss += loss.item() * len(y_b)
                n_samples += len(y_b)
                preds.extend(pred.cpu().numpy().ravel())
                trues.extend(y_b.cpu().numpy().ravel())
                
        avg_loss = running_loss / max(n_samples, 1)
        metrics = compute_all_metrics(np.array(trues), np.array(preds))
        
        print(
            f"[Round {server_round:02d}] Global Model Eval -> "
            f"Loss: {avg_loss:.4f} | MAE: {metrics['MAE']:.2f} | "
            f"RMSE: {metrics['RMSE']:.2f} | NASA Score: {metrics['NASA_Score']:.1f}"
        )
        
        return float(avg_loss), {
            "round": server_round,
            "mae": float(metrics["MAE"]),
            "rmse": float(metrics["RMSE"]),
            "r2": float(metrics["R2"]),
            "nasa_score": float(metrics["NASA_Score"]),
        }

    return evaluate


def get_fedavg_strategy(
    eval_fn: Optional[Callable] = None,
    initial_parameters: Optional[fl.common.Parameters] = None,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 3,
    min_available_clients: int = 3,
) -> fl.server.strategy.FedAvg:
    """Configures FedAvg strategy with metric aggregation and evaluation."""
    return fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        initial_parameters=initial_parameters,
        evaluate_fn=eval_fn,
        fit_metrics_aggregation_fn=weighted_average_metrics,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
    )
