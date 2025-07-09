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
from .callbacks import (
    CallbackManager, EarlyStoppingCallback, CheckpointCallback, 
    LoggingCallback, SchedulerCallback, MetricsCallback, 
    GradientClippingCallback, TrainingCurvesCallback, PairwiseInputAnalysisCallback
)
from ..models.base_model import BaseModel
from ..utils.directory import DirectoryManager


class Trainer:
    """Main training orchestration class."""
    
    def __init__(
        self,
        model: BaseModel,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        directory_manager: Optional[DirectoryManager] = None
    ):
        """Initialize Trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            device: Device to train on
            directory_manager: Directory manager for organizing outputs
        """
        self.model = model
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger('pytorch_pipeline')
        
        # Initialize directory manager
        self.directory_manager = directory_manager
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize components
        self._initialize_training_components()
        self._initialize_tracking()
        self._initialize_callbacks()
        
        self.logger.info(f"Initialized trainer on device: {self.device}")
        if self.directory_manager:
            self.logger.info(f"Using directory manager: {self.directory_manager}")
    
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
        
        # Metrics calculator (regression only)
        self.metrics_calculator = MetricsCalculator('regression')
        
        # Gradient clipping
        self.grad_clip_value = training_config.get('gradient_clipping', None)
    
    def _initialize_tracking(self) -> None:
        """Initialize training tracking variables."""
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_score = float('inf')
        self.best_epoch = 0
    
    def _initialize_callbacks(self) -> None:
        """Initialize callbacks from configuration."""
        training_config = self.config.get('training', {})
        callbacks = []
        
        # Metrics callback (always included)
        metrics_callback = MetricsCallback()
        callbacks.append(metrics_callback)
        self.metrics_callback = metrics_callback  # Store reference for access
        
        # Early stopping callback
        early_stopping_config = training_config.get('early_stopping', {})
        if early_stopping_config.get('enabled', False):
            early_stopping_callback = EarlyStoppingCallback(
                patience=early_stopping_config.get('patience', 10),
                min_delta=early_stopping_config.get('min_delta', 0.0),
                mode=early_stopping_config.get('mode', 'min'),
                restore_best_weights=early_stopping_config.get('restore_best_weights', True),
                monitor=early_stopping_config.get('monitor', 'val_loss')
            )
            callbacks.append(early_stopping_callback)
        
        # Checkpoint callback
        checkpoint_config = training_config.get('checkpointing', {})
        if checkpoint_config.get('enabled', True):
            checkpoint_dir = checkpoint_config.get('checkpoint_dir', 'checkpoints')
            if self.directory_manager:
                checkpoint_dir = str(self.directory_manager.checkpoints_dir)
            
            checkpoint_callback = CheckpointCallback(
                checkpoint_dir=checkpoint_dir,
                max_checkpoints=checkpoint_config.get('max_checkpoints', 5),
                monitor=checkpoint_config.get('monitor', 'val_loss'),
                mode=checkpoint_config.get('mode', 'min'),
                save_best_only=checkpoint_config.get('save_best_only', True)
            )
            callbacks.append(checkpoint_callback)
        
        # Logging callback
        log_config = training_config.get('logging', {})
        if log_config.get('enabled', True):
            log_dir = log_config.get('log_dir', 'logs')
            if self.directory_manager:
                log_dir = str(self.directory_manager.logs_dir)
            
            logging_callback = LoggingCallback(
                log_dir=log_dir,
                experiment_name=log_config.get('experiment_name', 'default'),
                use_tensorboard=log_config.get('use_tensorboard', True),
                log_every_n_batches=log_config.get('log_every_n_batches', 10)
            )
            callbacks.append(logging_callback)
        
        # Scheduler callback
        if self.scheduler is not None:
            scheduler_config = training_config.get('scheduler', {})
            step_mode = scheduler_config.get('step_mode', 'epoch')
            scheduler_callback = SchedulerCallback(self.scheduler, step_mode)
            callbacks.append(scheduler_callback)
        
        # Gradient clipping callback
        if self.grad_clip_value is not None:
            gradient_clipping_callback = GradientClippingCallback(self.grad_clip_value)
            callbacks.append(gradient_clipping_callback)
        
        # Training curves callback
        curves_config = training_config.get('diagnostic_plotting_curves', {})
        if curves_config.get('enabled', True):
            # Get scaler from config if available
            scaler = None
            if 'scaler' in self.config:
                scaler = self.config['scaler']
            
            update_frequency = curves_config.get('update_frequency', 1)
            if update_frequency is None:
                update_frequency = None
            
            plot_dir = curves_config.get('plot_dir', 'plots')
            if self.directory_manager:
                plot_dir = str(self.directory_manager.plots_dir)
            
            training_curves_callback = TrainingCurvesCallback(
                plot_dir=plot_dir,
                num_validation_samples=curves_config.get('num_validation_samples', 5),
                update_frequency=update_frequency,
                scaler=scaler,
                wavelength_range=curves_config.get('wavelength_range', None),
                seed=curves_config.get('seed', 42)
            )
            callbacks.append(training_curves_callback)
        
        # Pairwise input analysis callback
        pairplot_config = training_config.get('diagnostic_plotting_pairplot', {})
        if pairplot_config.get('enabled', True):
            # Get input scaler from config if available
            input_scaler = None
            if 'input_scaler' in self.config:
                input_scaler = self.config['input_scaler']
            
            update_frequency = pairplot_config.get('update_frequency', 10)
            if update_frequency is None:
                update_frequency = None
            
            plot_dir = pairplot_config.get('plot_dir', 'plots')
            if self.directory_manager:
                plot_dir = str(self.directory_manager.plots_dir)
            
            pairwise_analysis_callback = PairwiseInputAnalysisCallback(
                plot_dir=plot_dir,
                update_frequency=update_frequency,
                num_bins=pairplot_config.get('num_bins', 20),
                input_scaler=input_scaler
            )
            callbacks.append(pairwise_analysis_callback)
        
        # Initialize callback manager
        self.callback_manager = CallbackManager(callbacks)
    
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
        
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Callback: batch start
            self.callback_manager.on_batch_start(self, batch_idx, (inputs, targets))
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Callback: batch end (includes gradient clipping)
            self.callback_manager.on_batch_end(self, batch_idx, (inputs, targets), loss.item(), outputs)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            all_predictions.append(outputs.detach())
            all_targets.append(targets.detach())
            
            self.global_step += 1
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.metrics_calculator.calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = avg_loss  # Add loss to metrics
        
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
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                
                total_loss += loss.item()
                all_predictions.append(outputs.detach())
                all_targets.append(targets.detach())
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(val_loader)
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.metrics_calculator.calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = avg_loss  # Add loss to metrics
        
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
        # Store data loaders for callbacks
        self.train_loader = train_loader
        
        # Handle validation: if no val_loader provided, use train_loader for validation (overfitting check)
        if val_loader is None or len(val_loader.dataset) == 0:
            self.logger.info("No validation data provided, using training data for validation (overfitting check)")
            val_loader = train_loader
        self.val_loader = val_loader
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from_checkpoint:
            # Find checkpoint callback and load checkpoint
            for callback in self.callback_manager.callbacks:
                if isinstance(callback, CheckpointCallback):
                    checkpoint_info = callback.checkpoint_manager.load_checkpoint(
                        resume_from_checkpoint, self.model, self.optimizer, self.scheduler, self.device
                    )
                    start_epoch = checkpoint_info['epoch'] + 1
                    self.logger.info(f"Resumed training from epoch {start_epoch}")
                    break
        
        self.logger.info(f"Starting training for {epochs} epochs on {self.device}")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Optimizer: {self.optimizer.__class__.__name__}")
        if self.scheduler:
            self.logger.info(f"Scheduler: {self.scheduler.__class__.__name__}")
        
        # Callback: training start
        self.callback_manager.on_training_start(self)
        
        training_start_time = time.time()
        
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Callback: epoch start
            self.callback_manager.on_epoch_start(self, epoch)
            
            # Training phase
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_metrics = None, {}
            if val_loader is not None:
                val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # Callback: epoch end
            self.callback_manager.on_epoch_end(self, epoch, train_metrics, val_metrics)
            
            # Check if training should stop
            if self.callback_manager.should_stop_training():
                self.logger.info(f"Training stopped early at epoch {epoch}")
                break
        
        total_training_time = time.time() - training_start_time
        self.logger.info(f"Training completed in {total_training_time:.2f} seconds")
        
        # Callback: training end
        self.callback_manager.on_training_end(self)
        
        return {
            'history': self.metrics_callback.train_history,
            'train_metrics_history': self.metrics_callback.train_metrics_history,
            'val_metrics_history': self.metrics_callback.val_metrics_history,
            'best_val_score': self.best_val_score,
            'best_epoch': self.best_epoch,
            'total_training_time': total_training_time
        }
    
    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Test metrics
        """
        if test_loader is None or len(test_loader.dataset) == 0:
            self.logger.warning("No test data provided, skipping test evaluation")
            return {}
        
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
            'total_epochs': len(self.metrics_callback.train_history['epoch']),
            'final_train_loss': self.metrics_callback.train_history['train_loss'][-1] if self.metrics_callback.train_history['train_loss'] else None,
            'final_val_loss': self.metrics_callback.train_history['val_loss'][-1] if self.metrics_callback.train_history['val_loss'] else None
        } 