"""Main training orchestration module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, Any, Optional, Tuple, List
from torch.utils.data import DataLoader
import logging

from .metrics import MetricsCalculator
from .optimizers import OptimizerFactory, SchedulerFactory
from ..utils.checkpoints import CheckpointManager
from ..utils.logging import ExperimentLogger
from ..models.base_model import BaseModel


class EarlyStopping:
    """Early stopping utility to stop training when validation doesn't improve."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        """Initialize EarlyStopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for metrics where lower is better, 'max' for higher is better
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        self.best_state_dict = None
        
        self.logger = logging.getLogger('pytorch_pipeline')
    
    def __call__(self, score: float, model: nn.Module, epoch: int) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation score
            model: Model to save state from
            epoch: Current epoch
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_state_dict = model.state_dict().copy()
        
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights:
                self.best_state_dict = model.state_dict().copy()
        
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
            self.logger.info(f"Best score: {self.best_score:.6f} at epoch {self.best_epoch}")
            
            if self.restore_best_weights and self.best_state_dict is not None:
                model.load_state_dict(self.best_state_dict)
                self.logger.info("Restored best model weights")
            
            return True
        
        return False
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == 'min':
            return current < (best - self.min_delta)
        else:
            return current > (best + self.min_delta)


class Trainer:
    """Main training orchestration class."""
    
    def __init__(
        self,
        model: BaseModel,
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        """Initialize Trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger('pytorch_pipeline')
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize components
        self._initialize_training_components()
        self._initialize_tracking()
        
        self.logger.info(f"Initialized trainer on device: {self.device}")
    
    def _initialize_training_components(self) -> None:
        """Initialize training components from configuration."""
        training_config = self.config.get('training', {})
        
        # Loss function
        loss_config = training_config.get('loss', {'type': 'mse'})
        self.loss_fn = self._create_loss_function(loss_config)
        
        # Optimizer and scheduler
        optimizer_config = training_config.get('optimizer', {'type': 'adam', 'learning_rate': 1e-3})
        scheduler_config = training_config.get('scheduler', None)
        
        optimizer_factory = OptimizerFactory()
        scheduler_factory = SchedulerFactory()
        
        self.optimizer = optimizer_factory.create_optimizer(
            self.model.parameters(), optimizer_config
        )
        self.scheduler = None
        if scheduler_config and scheduler_config.get('type') != 'none':
            self.scheduler = scheduler_factory.create_scheduler(
                self.optimizer, scheduler_config
            )
        
        # Metrics calculator
        task_type = training_config.get('task_type', 'regression')
        num_classes = training_config.get('num_classes', None)
        self.metrics_calculator = MetricsCalculator(task_type, num_classes)
        
        # Early stopping
        early_stopping_config = training_config.get('early_stopping', {})
        if early_stopping_config.get('enabled', False):
            self.early_stopping = EarlyStopping(
                patience=early_stopping_config.get('patience', 10),
                min_delta=early_stopping_config.get('min_delta', 0.0),
                mode=early_stopping_config.get('mode', 'min'),
                restore_best_weights=early_stopping_config.get('restore_best_weights', True)
            )
        else:
            self.early_stopping = None
        
        # Gradient clipping
        self.grad_clip_value = training_config.get('gradient_clipping', None)
        
        # Checkpoint manager
        checkpoint_config = training_config.get('checkpointing', {})
        if checkpoint_config.get('enabled', True):
            checkpoint_dir = checkpoint_config.get('checkpoint_dir', 'checkpoints')
            max_checkpoints = checkpoint_config.get('max_checkpoints', 5)
            self.checkpoint_manager = CheckpointManager(checkpoint_dir, max_checkpoints)
        else:
            self.checkpoint_manager = None
        
        # Logging
        log_config = training_config.get('logging', {})
        if log_config.get('enabled', True):
            log_dir = log_config.get('log_dir', 'logs')
            experiment_name = log_config.get('experiment_name', 'default')
            use_tensorboard = log_config.get('use_tensorboard', True)
            self.experiment_logger = ExperimentLogger(
                log_dir, experiment_name, use_tensorboard=use_tensorboard
            )
        else:
            self.experiment_logger = None
    
    def _initialize_tracking(self) -> None:
        """Initialize training tracking variables."""
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_score = float('inf') if self.metrics_calculator.task_type == 'regression' else 0.0
        self.best_epoch = 0
        
        # Training history
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        self.train_metrics_history = {}
        self.val_metrics_history = {}
    
    def _create_loss_function(self, loss_config: Dict[str, Any]) -> nn.Module:
        """Create loss function from configuration.
        
        Args:
            loss_config: Loss function configuration
            
        Returns:
            Loss function module
        """
        loss_type = loss_config.get('type', 'mse').lower()
        
        if loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'mae' or loss_type == 'l1':
            return nn.L1Loss()
        elif loss_type == 'huber':
            delta = loss_config.get('delta', 1.0)
            return nn.HuberLoss(delta=delta)
        elif loss_type == 'cross_entropy':
            weight = loss_config.get('weight', None)
            ignore_index = loss_config.get('ignore_index', -100)
            return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        elif loss_type == 'bce':
            weight = loss_config.get('weight', None)
            return nn.BCELoss(weight=weight)
        elif loss_type == 'bce_with_logits':
            weight = loss_config.get('weight', None)
            pos_weight = loss_config.get('pos_weight', None)
            return nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, metrics)
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            
            self.optimizer.step()
            
            # Update scheduler if step-based
            if self.scheduler and hasattr(self.scheduler, 'step') and \
               getattr(self.scheduler, '_step_count', 0) is not None:
                # This is a step-based scheduler
                if hasattr(self.scheduler, 'step_size_up'):  # CyclicLR
                    self.scheduler.step()
                elif hasattr(self.scheduler, 'total_steps'):  # OneCycleLR
                    self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Store predictions for metrics calculation
            if self.metrics_calculator.task_type == 'regression':
                all_predictions.append(outputs.detach())
                all_targets.append(targets.detach())
            else:
                # For classification, get probabilities and predictions
                if self.metrics_calculator.task_type == 'binary_classification':
                    probs = torch.sigmoid(outputs).detach()
                    preds = (probs > 0.5).float()
                else:
                    probs = torch.softmax(outputs, dim=1).detach()
                    preds = torch.argmax(outputs, dim=1).detach()
                
                all_predictions.append(preds)
                all_targets.append(targets.detach())
                all_probabilities.append(probs)
            
            self.global_step += 1
            
            # Log batch-level metrics occasionally
            if batch_idx % max(1, len(train_loader) // 10) == 0:
                batch_time = time.time() - start_time
                self.logger.debug(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(train_loader)}: "
                    f"Loss = {loss.item():.6f}, Time = {batch_time:.2f}s"
                )
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        if all_probabilities:
            all_probabilities = torch.cat(all_probabilities, dim=0)
            metrics = self.metrics_calculator.calculate_metrics(
                all_predictions, all_targets, all_probabilities
            )
        else:
            metrics = self.metrics_calculator.calculate_metrics(
                all_predictions, all_targets
            )
        
        return avg_loss, metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, metrics)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                
                total_loss += loss.item()
                
                # Store predictions for metrics calculation
                if self.metrics_calculator.task_type == 'regression':
                    all_predictions.append(outputs.detach())
                    all_targets.append(targets.detach())
                else:
                    # For classification, get probabilities and predictions
                    if self.metrics_calculator.task_type == 'binary_classification':
                        probs = torch.sigmoid(outputs).detach()
                        preds = (probs > 0.5).float()
                    else:
                        probs = torch.softmax(outputs, dim=1).detach()
                        preds = torch.argmax(outputs, dim=1).detach()
                    
                    all_predictions.append(preds)
                    all_targets.append(targets.detach())
                    all_probabilities.append(probs)
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(val_loader)
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        if all_probabilities:
            all_probabilities = torch.cat(all_probabilities, dim=0)
            metrics = self.metrics_calculator.calculate_metrics(
                all_predictions, all_targets, all_probabilities
            )
        else:
            metrics = self.metrics_calculator.calculate_metrics(
                all_predictions, all_targets
            )
        
        return avg_loss, metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training history dictionary
        """
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from_checkpoint and self.checkpoint_manager:
            checkpoint_info = self.checkpoint_manager.load_checkpoint(
                resume_from_checkpoint, self.model, self.optimizer, self.scheduler, self.device
            )
            start_epoch = checkpoint_info['epoch'] + 1
            self.logger.info(f"Resumed training from epoch {start_epoch}")
        
        self.logger.info(f"Starting training for {epochs} epochs on {self.device}")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Optimizer: {self.optimizer.__class__.__name__}")
        if self.scheduler:
            self.logger.info(f"Scheduler: {self.scheduler.__class__.__name__}")
        
        training_start_time = time.time()
        
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_metrics = None, {}
            if val_loader is not None:
                val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # Update scheduler (epoch-based)
            if self.scheduler and not hasattr(self.scheduler, 'step_size_up') and \
               not hasattr(self.scheduler, 'total_steps'):
                if hasattr(self.scheduler, 'mode'):  # ReduceLROnPlateau
                    if val_loss is not None:
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Update tracking
            current_lr = self.optimizer.param_groups[0]['lr']
            self._update_history(epoch, train_loss, val_loss, current_lr, train_metrics, val_metrics)
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            self._log_epoch_results(epoch, train_loss, val_loss, train_metrics, val_metrics, epoch_time)
            
            # Checkpointing
            is_best = False
            if val_loader is not None:
                val_score = self.metrics_calculator.get_primary_metric(val_metrics)
                is_best = self.metrics_calculator.is_better_metric(val_score, self.best_val_score)
                if is_best:
                    self.best_val_score = val_score
                    self.best_epoch = epoch
            
            if self.checkpoint_manager:
                checkpoint_metrics = val_metrics if val_metrics else train_metrics
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, self.scheduler, epoch, checkpoint_metrics, is_best
                )
            
            # Early stopping
            if self.early_stopping and val_loader is not None:
                val_score = self.metrics_calculator.get_primary_metric(val_metrics)
                if self.early_stopping(val_score, self.model, epoch):
                    self.logger.info(f"Training stopped early at epoch {epoch}")
                    break
        
        total_training_time = time.time() - training_start_time
        self.logger.info(f"Training completed in {total_training_time:.2f} seconds")
        
        # Close logging
        if self.experiment_logger:
            self.experiment_logger.close()
        
        return {
            'history': self.train_history,
            'train_metrics_history': self.train_metrics_history,
            'val_metrics_history': self.val_metrics_history,
            'best_val_score': self.best_val_score,
            'best_epoch': self.best_epoch,
            'total_training_time': total_training_time
        }
    
    def _update_history(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float],
        learning_rate: float,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> None:
        """Update training history."""
        self.train_history['epoch'].append(epoch)
        self.train_history['train_loss'].append(train_loss)
        self.train_history['val_loss'].append(val_loss)
        self.train_history['learning_rate'].append(learning_rate)
        
        # Update metrics history
        for key, value in train_metrics.items():
            if key not in self.train_metrics_history:
                self.train_metrics_history[key] = []
            self.train_metrics_history[key].append(value)
        
        for key, value in val_metrics.items():
            if key not in self.val_metrics_history:
                self.val_metrics_history[key] = []
            self.val_metrics_history[key].append(value)
    
    def _log_epoch_results(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float],
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_time: float
    ) -> None:
        """Log epoch results."""
        # Console logging
        log_msg = f"Epoch {epoch:4d}: Train Loss = {train_loss:.6f}"
        if val_loss is not None:
            log_msg += f", Val Loss = {val_loss:.6f}"
        log_msg += f", LR = {self.optimizer.param_groups[0]['lr']:.2e}, Time = {epoch_time:.2f}s"
        self.logger.info(log_msg)
        
        # Log primary metrics
        if train_metrics:
            primary_train = self.metrics_calculator.get_primary_metric(train_metrics)
            metric_name = 'RMSE' if self.metrics_calculator.task_type == 'regression' else 'F1'
            log_msg = f"Epoch {epoch:4d}: Train {metric_name} = {primary_train:.6f}"
            if val_metrics:
                primary_val = self.metrics_calculator.get_primary_metric(val_metrics)
                log_msg += f", Val {metric_name} = {primary_val:.6f}"
            self.logger.info(log_msg)
        
        # TensorBoard logging
        if self.experiment_logger:
            self.experiment_logger.log_training_step(epoch, self.global_step, train_loss, train_metrics)
            if val_loss is not None:
                self.experiment_logger.log_validation_results(epoch, val_loss, val_metrics)
    
    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Test metrics
        """
        self.logger.info("Evaluating on test set...")
        test_loss, test_metrics = self.validate_epoch(test_loader)
        
        self.logger.info(f"Test Loss: {test_loss:.6f}")
        self.logger.info(f"Test Metrics: {self.metrics_calculator.format_metrics(test_metrics)}")
        
        return {
            'test_loss': test_loss,
            **test_metrics
        }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary.
        
        Returns:
            Training summary dictionary
        """
        return {
            'model_summary': self.model.get_model_summary(),
            'training_config': self.config.get('training', {}),
            'best_val_score': self.best_val_score,
            'best_epoch': self.best_epoch,
            'total_epochs': len(self.train_history['epoch']),
            'final_train_loss': self.train_history['train_loss'][-1] if self.train_history['train_loss'] else None,
            'final_val_loss': self.train_history['val_loss'][-1] if self.train_history['val_loss'] else None
        } 