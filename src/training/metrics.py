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
        
        This function calculates metrics assuming data shape (samples, wavelengths).
        For TARDIS emulation, each sample is a spectrum with multiple wavelength points.
        
        Args:
            predictions: Model predictions (Y_pred) with shape (samples, wavelengths)
            targets: Ground truth targets (Y) with shape (samples, wavelengths)
            
        Returns:
            Dictionary of regression metrics:
            - mse: Mean Squared Error averaged across all samples and wavelengths
            - mae: Mean Absolute Error averaged across all samples and wavelengths  
            - mae_mean_samples: Mean of MAE per sample (average over wavelengths, then mean over samples)
            - mae_max_samples: Maximum of MAE per sample (average over wavelengths, then max over samples)
        """
        # Convert to numpy for calculations
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Validate shapes
        if pred_np.shape != target_np.shape:
            raise ValueError(f"Predictions and targets must have same shape. Got {pred_np.shape} vs {target_np.shape}")
        
        if len(pred_np.shape) != 2:
            raise ValueError(f"Expected 2D arrays with shape (samples, wavelengths). Got shape {pred_np.shape}")
        
        # Mean Squared Error (Y-Y_pred)^2 - averaged across all samples and wavelengths
        mse = np.mean((target_np - pred_np) ** 2)
        
        # Mean Absolute Error - averaged across all samples and wavelengths
        mae = np.mean(np.abs(pred_np - target_np))
        
        # Calculate MAE per sample (average over wavelengths for each sample)
        mae_per_sample = np.mean(np.abs(pred_np - target_np), axis=1)  # Shape: (samples,)
        
        # Mean of MAE per sample
        mae_mean_samples = np.mean(mae_per_sample)
        
        # Maximum of MAE per sample  
        mae_max_samples = np.max(mae_per_sample)
        
        return {
            'mse': float(mse),                          # Global MSE across all samples and wavelengths
            'mae': float(mae),                          # Global MAE across all samples and wavelengths
            'mae_mean_samples': float(mae_mean_samples), # Mean of per-sample MAE (avg over wavelengths, then mean over samples)
            'mae_max_samples': float(mae_max_samples)   # Max of per-sample MAE (avg over wavelengths, then max over samples)
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
        
        For regression tasks, this is MSE.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Primary metric value
        """
        return metrics.get('mse', float('inf'))
    
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