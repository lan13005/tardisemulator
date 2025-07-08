"""Evaluation metrics for training and validation."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Union
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging


class MetricsCalculator:
    """Calculate various evaluation metrics for model performance."""
    
    def __init__(self, task_type: str = 'regression', num_classes: Optional[int] = None):
        """Initialize MetricsCalculator.
        
        Args:
            task_type: Type of task ('regression', 'classification', 'binary_classification')
            num_classes: Number of classes for classification tasks
        """
        self.task_type = task_type
        self.num_classes = num_classes
        self.logger = logging.getLogger('pytorch_pipeline')
        
        # Validate task type
        valid_tasks = ['regression', 'classification', 'binary_classification']
        if task_type not in valid_tasks:
            raise ValueError(f"task_type must be one of {valid_tasks}")
        
        if task_type in ['classification', 'binary_classification'] and num_classes is None:
            raise ValueError("num_classes must be specified for classification tasks")
    
    def calculate_regression_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate regression metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of regression metrics
        """
        # Convert to numpy for calculations
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Mean Squared Error
        mse = np.mean((pred_np - target_np) ** 2)
        
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
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(np.mean(mape)),
            'r2': float(r2),
            'explained_variance': float(explained_var),
            'max_error': float(max_error)
        }
    
    def calculate_classification_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Calculate classification metrics.
        
        Args:
            predictions: Model predictions (class indices)
            targets: Ground truth targets (class indices)
            probabilities: Class probabilities (optional)
            
        Returns:
            Dictionary of classification metrics
        """
        # Convert to numpy
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Accuracy
        accuracy = accuracy_score(target_np, pred_np)
        
        # Precision, Recall, F1-score
        precision, recall, f1, support = precision_recall_fscore_support(
            target_np, pred_np, average='weighted', zero_division=0
        )
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            target_np, pred_np, average='macro', zero_division=0
        )
        
        # Micro averages
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            target_np, pred_np, average='micro', zero_division=0
        )
        
        metrics = {
            'accuracy': float(accuracy),
            'precision_weighted': float(precision),
            'recall_weighted': float(recall),
            'f1_weighted': float(f1),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_micro': float(precision_micro),
            'recall_micro': float(recall_micro),
            'f1_micro': float(f1_micro)
        }
        
        # Add per-class metrics if not too many classes
        if self.num_classes <= 10:
            per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
                target_np, pred_np, average=None, zero_division=0
            )
            
            for i in range(len(per_class_precision)):
                metrics[f'precision_class_{i}'] = float(per_class_precision[i])
                metrics[f'recall_class_{i}'] = float(per_class_recall[i])
                metrics[f'f1_class_{i}'] = float(per_class_f1[i])
        
        # Top-k accuracy (if probabilities provided)
        if probabilities is not None:
            prob_np = probabilities.detach().cpu().numpy()
            for k in [1, 3, 5]:
                if k <= self.num_classes:
                    top_k_acc = self._calculate_top_k_accuracy(prob_np, target_np, k)
                    metrics[f'top_{k}_accuracy'] = float(top_k_acc)
        
        return metrics
    
    def calculate_binary_classification_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Calculate binary classification metrics.
        
        Args:
            predictions: Model predictions (0 or 1)
            targets: Ground truth targets (0 or 1)
            probabilities: Class probabilities (optional)
            threshold: Classification threshold
            
        Returns:
            Dictionary of binary classification metrics
        """
        # Convert to numpy
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Basic metrics
        tn, fp, fn, tp = confusion_matrix(target_np, pred_np, labels=[0, 1]).ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Matthews Correlation Coefficient
        mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1': float(f1),
            'mcc': float(mcc),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
        
        # AUC-ROC and AUC-PR (if probabilities provided)
        if probabilities is not None:
            try:
                from sklearn.metrics import roc_auc_score, average_precision_score
                prob_np = probabilities.detach().cpu().numpy()
                
                # Handle case where all targets are the same class
                if len(np.unique(target_np)) > 1:
                    auc_roc = roc_auc_score(target_np, prob_np)
                    auc_pr = average_precision_score(target_np, prob_np)
                    
                    metrics['auc_roc'] = float(auc_roc)
                    metrics['auc_pr'] = float(auc_pr)
                else:
                    metrics['auc_roc'] = 0.5
                    metrics['auc_pr'] = float(np.mean(target_np))
            
            except ImportError:
                self.logger.warning("sklearn not available for AUC calculations")
        
        return metrics
    
    def _calculate_top_k_accuracy(self, probabilities: np.ndarray, targets: np.ndarray, k: int) -> float:
        """Calculate top-k accuracy.
        
        Args:
            probabilities: Class probabilities
            targets: Ground truth targets
            k: Number of top predictions to consider
            
        Returns:
            Top-k accuracy
        """
        top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
        correct = 0
        total = len(targets)
        
        for i, target in enumerate(targets):
            if target in top_k_preds[i]:
                correct += 1
        
        return correct / total if total > 0 else 0
    
    def calculate_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate metrics based on task type.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            probabilities: Class probabilities (for classification)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of metrics
        """
        if self.task_type == 'regression':
            return self.calculate_regression_metrics(predictions, targets)
        elif self.task_type == 'classification':
            return self.calculate_classification_metrics(predictions, targets, probabilities)
        elif self.task_type == 'binary_classification':
            threshold = kwargs.get('threshold', 0.5)
            return self.calculate_binary_classification_metrics(
                predictions, targets, probabilities, threshold
            )
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def get_confusion_matrix(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> np.ndarray:
        """Get confusion matrix for classification tasks.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Confusion matrix as numpy array
        """
        if self.task_type == 'regression':
            raise ValueError("Confusion matrix not applicable for regression tasks")
        
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        labels = list(range(self.num_classes)) if self.num_classes else None
        return confusion_matrix(target_np, pred_np, labels=labels)
    
    def format_metrics(self, metrics: Dict[str, float], precision: int = 6) -> str:
        """Format metrics for display.
        
        Args:
            metrics: Dictionary of metrics
            precision: Number of decimal places
            
        Returns:
            Formatted string
        """
        formatted = []
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted.append(f"{key}: {value:.{precision}f}")
            else:
                formatted.append(f"{key}: {value}")
        
        return ", ".join(formatted)
    
    def get_primary_metric(self, metrics: Dict[str, float]) -> float:
        """Get the primary metric for model selection.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Primary metric value
        """
        if self.task_type == 'regression':
            return metrics.get('rmse', metrics.get('mse', 0))
        elif self.task_type in ['classification', 'binary_classification']:
            return metrics.get('f1_weighted', metrics.get('f1', metrics.get('accuracy', 0)))
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def is_better_metric(self, current: float, best: float) -> bool:
        """Check if current metric is better than best metric.
        
        Args:
            current: Current metric value
            best: Best metric value so far
            
        Returns:
            True if current is better than best
        """
        if self.task_type == 'regression':
            # Lower is better for regression (RMSE, MSE)
            return current < best
        else:
            # Higher is better for classification (accuracy, F1)
            return current > best 