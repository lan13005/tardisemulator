"""Evaluation metrics for training and validation - focused on regression tasks for TARDIS emulator."""

import torch
import numpy as np
from typing import Dict, Any, Optional
import logging


class MetricsCalculator:
    """Calculate evaluation metrics for model performance - optimized for regression tasks."""
    
    def __init__(self, task_type: str = 'regression'):
        """Initialize MetricsCalculator.
        
        Args:
            task_type: Type of task (only 'regression' supported)
        """
        self.task_type = task_type
        self.logger = logging.getLogger('pytorch_pipeline')
        
        # Validate task type
        if task_type != 'regression':
            raise ValueError("Only 'regression' task type is supported")
    
    def calculate_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate comprehensive regression metrics.
        
        This is the primary function for TARDIS spectrum emulation tasks.
        
        Args:
            predictions: Model predictions (Y_pred)
            targets: Ground truth targets (Y)
            
        Returns:
            Dictionary of regression metrics
        """
        # Convert to numpy for calculations
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Mean Squared Error (Y-Y_pred)^2 - Primary metric for TARDIS emulation
        mse = np.mean((target_np - pred_np) ** 2)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(pred_np - target_np))
        
        # Mean Absolute Percentage Error (handle division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((target_np - pred_np) / target_np)) * 100
            mape = np.where(np.isfinite(mape), mape, 0)
        
        # R-squared (coefficient of determination)
        ss_res = np.sum((target_np - pred_np) ** 2)
        ss_tot = np.sum((target_np - np.mean(target_np)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Explained Variance Score
        var_y = np.var(target_np)
        explained_var = 1 - np.var(target_np - pred_np) / var_y if var_y != 0 else 0
        
        # Max Error
        max_error = np.max(np.abs(pred_np - target_np))
        
        return {
            'mse': float(mse),                          # Primary metric: (Y-Y_pred)^2
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(np.mean(mape)),
            'r2': float(r2),
            'explained_variance': float(explained_var),
            'max_error': float(max_error)
        }
    
    def format_metrics(self, metrics: Dict[str, float], precision: int = 6) -> str:
        """Format metrics for logging.
        
        Args:
            metrics: Dictionary of metrics
            precision: Number of decimal places
            
        Returns:
            Formatted string of metrics
        """
        formatted_parts = []
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted_parts.append(f"{key}: {value:.{precision}f}")
            else:
                formatted_parts.append(f"{key}: {value}")
        
        return ", ".join(formatted_parts)
    
    def get_primary_metric(self, metrics: Dict[str, float]) -> float:
        """Get the primary metric for model evaluation.
        
        For regression tasks, this is RMSE.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Primary metric value
        """
        return metrics.get('rmse', float('inf'))
    
    def is_better_metric(self, current: float, best: float) -> bool:
        """Check if current metric is better than best metric.
        
        For regression tasks, lower is better.
        
        Args:
            current: Current metric value
            best: Best metric value so far
            
        Returns:
            True if current is better than best
        """
        return current < best 