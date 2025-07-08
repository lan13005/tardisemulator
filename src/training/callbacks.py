"""Training callbacks for modular training orchestration."""

import torch
import torch.nn as nn
import numpy as np
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import os

from .metrics import MetricsCalculator
from ..utils.checkpoints import CheckpointManager
from ..utils.logging import ExperimentLogger


class Callback(ABC):
    """Base callback class for training hooks."""
    
    def on_training_start(self, trainer) -> None:
        """Called at the start of training."""
        pass
    
    def on_epoch_start(self, trainer, epoch: int) -> None:
        """Called at the start of each epoch."""
        pass
    
    def on_batch_start(self, trainer, batch_idx: int, batch: Tuple) -> None:
        """Called at the start of each batch."""
        pass
    
    def on_batch_end(self, trainer, batch_idx: int, batch: Tuple, loss: float, outputs: torch.Tensor) -> None:
        """Called at the end of each batch."""
        pass
    
    def on_epoch_end(self, trainer, epoch: int, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]]) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_training_end(self, trainer) -> None:
        """Called at the end of training."""
        pass


class EarlyStoppingCallback(Callback):
    """Early stopping callback to stop training when validation doesn't improve."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True,
        monitor: str = 'val_loss'
    ):
        """Initialize EarlyStoppingCallback.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for metrics where lower is better, 'max' for higher is better
            restore_best_weights: Whether to restore best weights when stopping
            monitor: Metric to monitor for early stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor
        
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        self.best_state_dict = None
        self.should_stop = False
        
        self.logger = logging.getLogger('pytorch_pipeline')
    
    def on_epoch_end(self, trainer, epoch: int, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]]) -> None:
        """Check if training should stop."""
        if val_metrics is None or self.monitor not in val_metrics:
            return
        
        score = val_metrics[self.monitor]
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_state_dict = trainer.model.state_dict().copy()
        
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights:
                self.best_state_dict = trainer.model.state_dict().copy()
        
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
            self.logger.info(f"Best {self.monitor}: {self.best_score:.6f} at epoch {self.best_epoch}")
            
            if self.restore_best_weights and self.best_state_dict is not None:
                trainer.model.load_state_dict(self.best_state_dict)
                self.logger.info("Restored best model weights")
            
            self.should_stop = True
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == 'min':
            return current < (best - self.min_delta)
        else:
            return current > (best + self.min_delta)


class CheckpointCallback(Callback):
    """Checkpoint callback for saving model checkpoints."""
    
    def __init__(
        self,
        checkpoint_dir: str = 'checkpoints',
        max_checkpoints: int = 5,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True
    ):
        """Initialize CheckpointCallback.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            monitor: Metric to monitor for best checkpoint
            mode: 'min' for metrics where lower is better, 'max' for higher is better
            save_best_only: Whether to save only the best checkpoint
        """
        self.checkpoint_manager = CheckpointManager(checkpoint_dir, max_checkpoints)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        
        self.logger = logging.getLogger('pytorch_pipeline')
    
    def on_epoch_end(self, trainer, epoch: int, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]]) -> None:
        """Save checkpoint if conditions are met."""
        if val_metrics is None or self.monitor not in val_metrics:
            return
        
        current_score = val_metrics[self.monitor]
        is_best = False
        
        if self.mode == 'min':
            is_best = current_score < self.best_score
        else:
            is_best = current_score > self.best_score
        
        if is_best:
            self.best_score = current_score
        
        # Save checkpoint
        if not self.save_best_only or is_best:
            checkpoint_metrics = val_metrics if val_metrics else train_metrics
            self.checkpoint_manager.save_checkpoint(
                trainer.model, trainer.optimizer, trainer.scheduler, 
                epoch, checkpoint_metrics, is_best
            )


class LoggingCallback(Callback):
    """Logging callback for training metrics and TensorBoard logging."""
    
    def __init__(
        self,
        log_dir: str = 'logs',
        experiment_name: str = 'default',
        use_tensorboard: bool = True,
        log_every_n_batches: int = 10
    ):
        """Initialize LoggingCallback.
        
        Args:
            log_dir: Directory for logs
            experiment_name: Name of the experiment
            use_tensorboard: Whether to use TensorBoard logging
            log_every_n_batches: Log every N batches
        """
        self.experiment_logger = ExperimentLogger(log_dir, experiment_name, use_tensorboard=use_tensorboard)
        self.log_every_n_batches = log_every_n_batches
        self.logger = logging.getLogger('pytorch_pipeline')
    
    def on_batch_end(self, trainer, batch_idx: int, batch: Tuple, loss: float, outputs: torch.Tensor) -> None:
        """Log batch-level metrics."""
        if batch_idx % self.log_every_n_batches == 0:
            self.logger.debug(
                f"Epoch {trainer.current_epoch}, Batch {batch_idx}: Loss = {loss:.6f}"
            )
    
    def on_epoch_end(self, trainer, epoch: int, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]]) -> None:
        """Log epoch-level metrics."""
        # Console logging
        log_msg = f"Epoch {epoch:4d}: Train Loss = {train_metrics.get('loss', 0):.6f}"
        if val_metrics:
            log_msg += f", Val Loss = {val_metrics.get('loss', 0):.6f}"
        log_msg += f", LR = {trainer.optimizer.param_groups[0]['lr']:.2e}"
        self.logger.info(log_msg)
        
        # Log primary metrics
        if train_metrics:
            primary_train = train_metrics.get('rmse', train_metrics.get('loss', 0))
            log_msg = f"Epoch {epoch:4d}: Train RMSE = {primary_train:.6f}"
            if val_metrics:
                primary_val = val_metrics.get('rmse', val_metrics.get('loss', 0))
                log_msg += f", Val RMSE = {primary_val:.6f}"
            self.logger.info(log_msg)
        
        # TensorBoard logging
        self.experiment_logger.log_training_step(epoch, trainer.global_step, train_metrics.get('loss', 0), train_metrics)
        if val_metrics:
            self.experiment_logger.log_validation_results(epoch, val_metrics.get('loss', 0), val_metrics)
    
    def on_training_end(self, trainer) -> None:
        """Close logging."""
        self.experiment_logger.close()


class SchedulerCallback(Callback):
    """Scheduler callback for managing learning rate scheduling."""
    
    def __init__(self, scheduler, step_mode: str = 'epoch'):
        """Initialize SchedulerCallback.
        
        Args:
            scheduler: Learning rate scheduler
            step_mode: When to step scheduler ('epoch' or 'batch')
        """
        self.scheduler = scheduler
        self.step_mode = step_mode
    
    def on_batch_end(self, trainer, batch_idx: int, batch: Tuple, loss: float, outputs: torch.Tensor) -> None:
        """Step scheduler after each batch if step_mode is 'batch'."""
        if self.step_mode == 'batch' and self.scheduler is not None:
            # Check if it's a step-based scheduler
            if hasattr(self.scheduler, 'step_size_up') or hasattr(self.scheduler, 'total_steps'):
                self.scheduler.step()
    
    def on_epoch_end(self, trainer, epoch: int, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]]) -> None:
        """Step scheduler after each epoch if step_mode is 'epoch'."""
        if self.step_mode == 'epoch' and self.scheduler is not None:
            # Check if it's an epoch-based scheduler
            if not hasattr(self.scheduler, 'step_size_up') and not hasattr(self.scheduler, 'total_steps'):
                if hasattr(self.scheduler, 'mode'):  # ReduceLROnPlateau
                    if val_metrics is not None:
                        self.scheduler.step(val_metrics.get('loss', 0))
                else:
                    self.scheduler.step()


class MetricsCallback(Callback):
    """Metrics callback for tracking training history."""
    
    def __init__(self):
        """Initialize MetricsCallback."""
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        self.train_metrics_history = {}
        self.val_metrics_history = {}
    
    def on_epoch_end(self, trainer, epoch: int, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]]) -> None:
        """Update training history."""
        current_lr = trainer.optimizer.param_groups[0]['lr']
        
        self.train_history['epoch'].append(epoch)
        self.train_history['train_loss'].append(train_metrics.get('loss', 0))
        self.train_history['val_loss'].append(val_metrics.get('loss', 0) if val_metrics else None)
        self.train_history['learning_rate'].append(current_lr)
        
        # Update metrics history
        for key, value in train_metrics.items():
            if key not in self.train_metrics_history:
                self.train_metrics_history[key] = []
            self.train_metrics_history[key].append(value)
        
        if val_metrics:
            for key, value in val_metrics.items():
                if key not in self.val_metrics_history:
                    self.val_metrics_history[key] = []
                self.val_metrics_history[key].append(value)


class GradientClippingCallback(Callback):
    """Gradient clipping callback."""
    
    def __init__(self, max_norm: float = 1.0):
        """Initialize GradientClippingCallback.
        
        Args:
            max_norm: Maximum gradient norm
        """
        self.max_norm = max_norm
    
    def on_batch_end(self, trainer, batch_idx: int, batch: Tuple, loss: float, outputs: torch.Tensor) -> None:
        """Clip gradients after backward pass."""
        if self.max_norm is not None:
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), self.max_norm)


class TrainingCurvesCallback(Callback):
    """Training curves callback for training curves and validation predictions."""
    
    def __init__(
        self,
        plot_dir: str = 'plots',
        num_validation_samples: int = 5,
        update_frequency: int = 1,
        scaler=None,
        wavelength_range: Optional[Tuple[float, float]] = None,
        seed: int = 42
    ):
        """Initialize TrainingCurvesCallback.
        
        Args:
            plot_dir: Directory to save plots
            num_validation_samples: Number of validation samples to plot
            update_frequency: Update plots every N epochs
            scaler: Scaler for inverse transforming predictions
            wavelength_range: Range of wavelengths for plotting (min, max)
            seed: Random seed for selecting validation samples to track
        """
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(exist_ok=True)
        self.num_validation_samples = num_validation_samples
        self.update_frequency = update_frequency
        self.scaler = scaler
        self.wavelength_range = wavelength_range
        self.seed = seed
        
        self.logger = logging.getLogger('pytorch_pipeline')
        
        # Store validation data for plotting - same samples throughout training
        self.val_inputs = None
        self.val_targets = None
        self.val_predictions = None
        self.sample_indices = None  # Store which samples we're tracking
        
        # Initialize plots
        self.fig, self.axes = None, None
        self._setup_plots()
    
    def _setup_plots(self):
        """Setup matplotlib figures and axes with multiple subplots."""
        plt.style.use('default')
        
        # Create subplots: 1 for training curves + num_validation_samples for individual samples
        n_subplots = 1 + self.num_validation_samples
        self.fig, self.axes = plt.subplots(n_subplots, 1, figsize=(12, 3 * n_subplots))
        
        # Ensure axes is always a list/array for consistent indexing
        if n_subplots == 1:
            self.axes = [self.axes]
                
        # Training curves subplot (first subplot)
        self.axes[0].set_title('Training Curves')
        self.axes[0].set_xlabel('Epoch')
        self.axes[0].set_ylabel('Loss')
        self.axes[0].grid(True, alpha=0.3)
        
        # Individual validation sample subplots
        for i in range(self.num_validation_samples):
            self.axes[i + 1].set_title(f'Validation Sample {i + 1} (Truth vs Predicted)')
            self.axes[i + 1].set_xlabel('Wavelength (Angstroms)')
            self.axes[i + 1].set_ylabel('Flux')
            self.axes[i + 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def on_training_start(self, trainer) -> None:
        """Store validation data for plotting."""
        # Get validation data from trainer if available
        if hasattr(trainer, 'val_loader') and trainer.val_loader is not None:
            self._store_validation_data(trainer.val_loader, trainer)
    
    def on_epoch_end(self, trainer, epoch: int, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]]) -> None:
        """Update plots if it's time."""
        if self.update_frequency is None:
            return
        if epoch % self.update_frequency == 0:
            self._update_training_curves(trainer, epoch)
            if self.val_predictions is not None:
                self._update_validation_predictions(trainer, epoch)
            self._save_plots(epoch)
    
    def _store_validation_data(self, val_loader, trainer):
        """Store validation data for plotting - select random samples to track."""
        trainer.model.eval()
        all_inputs = []
        all_targets = []
        all_predictions = []
        
        # Collect all validation data first
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(trainer.device)
                targets = targets.to(trainer.device)
                
                outputs = trainer.model(inputs)
                
                all_inputs.append(inputs.cpu())
                all_targets.append(targets.cpu())
                all_predictions.append(outputs.cpu())
        
        if all_inputs:
            # Concatenate all batches
            all_inputs = torch.cat(all_inputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            all_predictions = torch.cat(all_predictions, dim=0)
            
            # Select random samples to track throughout training
            total_samples = len(all_targets)
            num_samples_to_track = min(self.num_validation_samples, total_samples)
            
            # Use fixed random seed for reproducible sample selection
            np.random.seed(self.seed)
            self.sample_indices = np.random.choice(total_samples, num_samples_to_track, replace=False)
            
            # Store the selected samples
            self.val_inputs = all_inputs[self.sample_indices]
            self.val_targets = all_targets[self.sample_indices]
            self.val_predictions = all_predictions[self.sample_indices]
            
            self.logger.info(f"Selected {num_samples_to_track} validation samples for tracking (indices: {self.sample_indices})")
    
    def _update_training_curves(self, trainer, epoch):
        """Update training curves plot (first subplot)."""
        if not hasattr(trainer, 'metrics_callback'):
            return
        
        history = trainer.metrics_callback.train_history
        
        self.axes[0].clear()
        
        # Get current average losses for title
        current_train_loss = history['train_loss'][-1] if history['train_loss'] else 0.0
        current_val_loss = history['val_loss'][-1] if history['val_loss'] and history['val_loss'][-1] is not None else None
        
        title = f'Training Curves (Train Loss: {current_train_loss:.6f}'
        if current_val_loss is not None:
            title += f', Val Loss: {current_val_loss:.6f}'
        title += ')'
        
        self.axes[0].set_title(title)
        self.axes[0].set_xlabel('Epoch')
        self.axes[0].set_ylabel('Loss')
        self.axes[0].grid(True, alpha=0.3)
        
        if history['epoch']:
            epochs = history['epoch']
            train_losses = history['train_loss']
            val_losses = history['val_loss']
            
            self.axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
            if any(loss is not None for loss in val_losses):
                self.axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
            
            self.axes[0].legend()
            self.axes[0].set_xlim(0, max(max(epochs), 1) if epochs else epoch)
    
    def _update_validation_predictions(self, trainer, epoch):
        """Update validation predictions plot (individual sample subplots)."""
        if self.val_predictions is None:
            return
        
        # Get latest predictions for the tracked samples
        trainer.model.eval()
        with torch.no_grad():
            latest_predictions = trainer.model(self.val_inputs.to(trainer.device)).cpu()
        
        # Update each sample subplot
        for i in range(len(self.val_targets)):
            if i >= self.num_validation_samples:
                break
                
            self.axes[i + 1].clear()
            self.axes[i + 1].set_title(f'Validation Sample {i + 1} (Epoch {epoch})')
            self.axes[i + 1].set_xlabel('Wavelength (Angstroms)')
            self.axes[i + 1].set_ylabel('Flux')
            self.axes[i + 1].grid(True, alpha=0.3)
            
            # Inverse transform if scaler is available
            if self.scaler is not None:
                target_original = self.scaler.inverse_transform(self.val_targets[i].numpy().reshape(1, -1)).flatten()
                pred_original = self.scaler.inverse_transform(latest_predictions[i].numpy().reshape(1, -1)).flatten()
            else:
                target_original = self.val_targets[i].numpy()
                pred_original = latest_predictions[i].numpy()
            
            # Create wavelength array
            if self.wavelength_range:
                wavelengths = np.linspace(self.wavelength_range[0], self.wavelength_range[1], len(target_original))
            else:
                wavelengths = np.arange(len(target_original))
            
            # Plot truth and prediction
            self.axes[i + 1].plot(wavelengths, target_original, 'b-', label='Truth', linewidth=2, alpha=0.8)
            self.axes[i + 1].plot(wavelengths, pred_original, 'r--', label='Prediction', linewidth=2, alpha=0.8)
            
            # Add sample index information
            if self.sample_indices is not None:
                self.axes[i + 1].text(0.02, 0.98, f'Sample Index: {self.sample_indices[i]}', 
                                    transform=self.axes[i + 1].transAxes, 
                                    verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            self.axes[i + 1].legend()
    
    def _save_plots(self, epoch):
        """Save plots to file."""
        plot_path = self.plot_dir / f'training_diagnostics_epoch_{epoch:04d}.png'
        self.fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Saved diagnostic plots to {plot_path}")
    
    def on_training_end(self, trainer) -> None:
        """Save final plots."""
        if self.fig is not None and self.update_frequency is not None:
            final_plot_path = self.plot_dir / 'final_training_diagnostics.png'
            self.fig.savefig(final_plot_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved final diagnostic plots to {final_plot_path}")
            plt.close(self.fig)


class PairwiseInputAnalysisCallback(Callback):
    """Pairwise input analysis callback for analyzing loss patterns across input parameter pairs."""
    
    def __init__(
        self,
        plot_dir: str = 'plots',
        update_frequency: int = 10,
        num_bins: int = 20,
        input_scaler=None
    ):
        """Initialize PairwiseInputAnalysisCallback.
        
        Args:
            plot_dir: Directory to save plots
            update_frequency: Update plots every N epochs (less frequent due to compute cost)
            num_bins: Number of bins for X and Y axes in the histogram/contour plot
            input_scaler: Scaler for inverse transforming inputs (if preprocessing was applied)
        """
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(exist_ok=True)
        self.update_frequency = update_frequency
        self.num_bins = num_bins
        self.input_scaler = input_scaler
        self._colorbars = {}
        
        self.logger = logging.getLogger('pytorch_pipeline')
        
        # Store validation data for analysis
        self.val_inputs = None
        self.val_targets = None
        self.val_predictions = None
        self.val_losses = None
        
        # Store feature names for proper labeling
        self.feature_names = None
        
        # Initialize plots
        self.fig, self.axes = None, None
        self._setup_plots()
    
    def _setup_plots(self):
        """Setup matplotlib figures and axes for pairwise analysis."""
        plt.style.use('default')
        
        # We'll determine the number of input parameters dynamically
        # For now, create a placeholder that will be updated when we have data
        self.fig = None
        self.axes = None
        self.n_input_params = None
    
    def on_training_start(self, trainer) -> None:
        """Store validation data for analysis."""
        # Get validation data from trainer if available
        if hasattr(trainer, 'val_loader') and trainer.val_loader is not None:
            self._store_validation_data(trainer.val_loader, trainer)
    
    def on_epoch_end(self, trainer, epoch: int, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]]) -> None:
        """Update plots if it's time."""
        if self.update_frequency is None:
            return
        if epoch % self.update_frequency == 0:
            self._update_pairwise_analysis(trainer, epoch)
            self._save_plots(epoch)
    
    def _store_validation_data(self, val_loader, trainer):
        """Store validation data for analysis."""
        trainer.model.eval()
        all_inputs = []
        all_targets = []
        all_predictions = []
        
        # Collect all validation data first
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(trainer.device)
                targets = targets.to(trainer.device)
                
                outputs = trainer.model(inputs)
                
                all_inputs.append(inputs.cpu())
                all_targets.append(targets.cpu())
                all_predictions.append(outputs.cpu())
        
        if all_inputs:
            # Concatenate all batches
            self.val_inputs = torch.cat(all_inputs, dim=0)
            self.val_targets = torch.cat(all_targets, dim=0)
            self.val_predictions = torch.cat(all_predictions, dim=0)
            
            # Get feature names from dataloader if available
            self.feature_names = self._get_feature_names(val_loader)
            
            # Determine number of input parameters and setup plots
            self.n_input_params = self.val_inputs.shape[1]
            self._create_plot_grid()
            
            self.logger.info(f"Stored {len(self.val_inputs)} validation samples for pairwise analysis with {self.n_input_params} input parameters")
            if self.feature_names:
                self.logger.info(f"Feature names: {self.feature_names}")
    
    def _get_feature_names(self, val_loader):
        """Get feature names from the dataloader."""
        try:
            # Try to get feature names from the dataset
            dataset = val_loader.dataset
            if hasattr(dataset, 'feature_names'):
                return dataset.feature_names
            
            # Try to get from dataloader's dataset's dataloader (if it's a HDF5DataLoader)
            if hasattr(dataset, 'dataloader') and hasattr(dataset.dataloader, 'feature_names'):
                return dataset.dataloader.feature_names
            
            # Try to get from trainer's dataloader
            if hasattr(val_loader, 'feature_names'):
                return val_loader.feature_names
                
        except Exception as e:
            self.logger.warning(f"Could not get feature names: {e}")
        
        return None
    
    def _get_param_name(self, param_idx):
        """Get parameter name for the given index."""
        if self.feature_names and param_idx < len(self.feature_names):
            # Clean up the feature name for display
            name = self.feature_names[param_idx]
            # Remove common prefixes and make it more readable
            if name.startswith('model.'):
                name = name[6:]  # Remove 'model.' prefix
            if name.startswith('supernova.'):
                name = name[10:]  # Remove 'supernova.' prefix
            if name.startswith('mass_'):
                name = name[5:]  # Remove 'mass_' prefix
            # Replace underscores with spaces and capitalize
            name = name.replace('_', ' ').title()
            
            # Truncate to 5-letter substrings if period or space delimited
            if '.' in name or ' ' in name:
                # Split by periods or spaces
                parts = name.replace('.', ' ').split()
                # Take first 5 letters of each part and concatenate
                truncated_parts = [part[:5] for part in parts]
                name = ''.join(truncated_parts)
            
            return name
        else:
            return f'Input {param_idx+1}'
    
    def _create_plot_grid(self):
        """Create the plot grid for pairwise analysis."""
        if self.n_input_params is None:
            return
            
        # Create a grid of subplots: n_input_params x n_input_params
        figsize = (3 * self.n_input_params, 3 * self.n_input_params)
        self.fig, self.axes = plt.subplots(self.n_input_params, self.n_input_params, figsize=figsize)
        
        # Ensure axes is always 2D for consistent indexing
        if self.n_input_params == 1:
            self.axes = np.array([[self.axes]])
        elif self.n_input_params == 2:
            self.axes = np.array(self.axes).reshape(2, 2)
                
        # Set up all subplots
        for i in range(self.n_input_params):
            for j in range(self.n_input_params):
                if i == j:
                    # Diagonal: 1D histogram of losses along this input dimension
                    param_name = self._get_param_name(i)
                    self.axes[i, j].set_title(f'{param_name} Loss Distribution')
                    self.axes[i, j].set_xlabel(param_name)
                    self.axes[i, j].set_ylabel('Count')
                elif i < j:
                    # Upper triangle: 2D histogram of average loss
                    param_i_name = self._get_param_name(i)
                    param_j_name = self._get_param_name(j)
                    self.axes[i, j].set_title(f'{param_i_name} vs {param_j_name}')
                    self.axes[i, j].set_xlabel(param_i_name)
                    self.axes[i, j].set_ylabel(param_j_name)
                else:
                    # Lower triangle: empty (or could be used for other plots)
                    self.axes[i, j].set_visible(False)
        
        plt.tight_layout()
    
    def _update_pairwise_analysis(self, trainer, epoch):
        """Update pairwise analysis plot."""
        if self.val_inputs is None or self.fig is None:
            return
        
        # Get latest predictions and calculate per-sample losses for the validation set
        trainer.model.eval()
        with torch.no_grad():
            latest_predictions = trainer.model(self.val_inputs.to(trainer.device)).cpu()
            # Calculate per-sample losses (MSE for each sample)
            latest_losses = torch.mean((latest_predictions - self.val_targets) ** 2, dim=1)
        
        # Inverse transform inputs if scaler is available
        if self.input_scaler is not None:
            inputs_original = self.input_scaler.inverse_transform(self.val_inputs.numpy())
        else:
            inputs_original = self.val_inputs.numpy()
        
        losses = latest_losses.numpy()
        
        # Calculate average loss for title
        avg_loss = np.mean(losses)
        
        # Update overall title first
        self.fig.suptitle(f'Pairwise Input Analysis: Loss Distribution (Epoch {epoch}, Avg Loss: {avg_loss:.6f})', fontsize=16)
        
        # Update all subplots
        for i in range(self.n_input_params):
            for j in range(self.n_input_params):
                if i == j:
                    # Diagonal: 1D histogram of losses along this input dimension
                    self._plot_1d_histogram(i, inputs_original[:, i], losses, epoch)
                elif i < j:
                    # Upper triangle: 2D histogram of average loss
                    self._plot_2d_histogram(i, j, inputs_original[:, i], inputs_original[:, j], losses, epoch)
    
    def _plot_1d_histogram(self, param_idx, param_values, losses, epoch):
        """Plot 1D histogram of losses along a single input parameter."""
        ax = self.axes[param_idx, param_idx]
        ax.clear()
        
        # Create bins for the parameter
        bins = np.linspace(param_values.min(), param_values.max(), self.num_bins + 1)
        
        # Create histogram with average loss in each bin
        hist, bin_edges = np.histogram(param_values, bins=bins)
        hist_loss, _ = np.histogram(param_values, bins=bins, weights=losses)
        
        # Calculate average loss in each bin
        avg_loss = np.divide(hist_loss, hist, out=np.zeros_like(hist_loss), where=hist != 0)
        
        # Plot histogram
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, avg_loss, width=bin_edges[1] - bin_edges[0], alpha=0.7, color='skyblue', edgecolor='black')
    
        param_name = self._get_param_name(param_idx)
        ax.set_title(f'{param_name} Loss Distribution')
        ax.set_xlabel(param_name)
        ax.set_ylabel('Average Loss')
        ax.grid(True, alpha=0.3)
    
    def _plot_2d_histogram(self, param_i, param_j, param_i_values, param_j_values, losses, epoch):
        """Plot 2D histogram of average loss for a pair of input parameters."""
        ax = self.axes[param_i, param_j]

        if (param_i, param_j) in self._colorbars:
            self._colorbars[(param_i, param_j)].remove()
            del self._colorbars[(param_i, param_j)]

        ax.clear()
        
        # Create bins
        x_bins = np.linspace(param_i_values.min(), param_i_values.max(), self.num_bins + 1)
        y_bins = np.linspace(param_j_values.min(), param_j_values.max(), self.num_bins + 1)
        
        # Create 2D histogram with average loss in each bin
        H, xedges, yedges = np.histogram2d(param_i_values, param_j_values, bins=[x_bins, y_bins])
        H_loss, _, _ = np.histogram2d(param_i_values, param_j_values, bins=[x_bins, y_bins], weights=losses)
        
        # Calculate average loss in each bin
        avg_loss = np.divide(H_loss, H, out=np.zeros_like(H_loss), where=H != 0)
        
        # Plot 2D histogram (imshow for better performance than contourf)
        im = ax.imshow(avg_loss.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                      aspect='auto', cmap='viridis')
        
        # Add colorbar (let matplotlib handle it automatically)
        cbar = plt.colorbar(im, ax=ax, label='Average Loss')
        self._colorbars[(param_i, param_j)] = cbar
        
        param_i_name = self._get_param_name(param_i)
        param_j_name = self._get_param_name(param_j)
        ax.set_title(f'{param_i_name} vs {param_j_name}')
        ax.set_xlabel(param_i_name)
        ax.set_ylabel(param_j_name)
        ax.grid(True, alpha=0.3)
    
    def _save_plots(self, epoch):
        """Save plots to file."""
        if self.fig is None:
            return
        plot_path = self.plot_dir / f'pairwise_analysis_epoch_{epoch:04d}.png'
        self.fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Saved pairwise analysis plots to {plot_path}")
    
    def on_training_end(self, trainer) -> None:
        """Save final plots."""
        if self.fig is not None and self.update_frequency is not None:
            final_plot_path = self.plot_dir / 'final_pairwise_analysis.png'
            self.fig.savefig(final_plot_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved final pairwise analysis plots to {final_plot_path}")
            plt.close(self.fig)


class CallbackManager:
    """Manager for organizing and executing callbacks."""
    
    def __init__(self, callbacks: List[Callback]):
        """Initialize CallbackManager.
        
        Args:
            callbacks: List of callbacks to manage
        """
        self.callbacks = callbacks
    
    def on_training_start(self, trainer) -> None:
        """Execute on_training_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_training_start(trainer)
    
    def on_epoch_start(self, trainer, epoch: int) -> None:
        """Execute on_epoch_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_start(trainer, epoch)
    
    def on_batch_start(self, trainer, batch_idx: int, batch: Tuple) -> None:
        """Execute on_batch_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_start(trainer, batch_idx, batch)
    
    def on_batch_end(self, trainer, batch_idx: int, batch: Tuple, loss: float, outputs: torch.Tensor) -> None:
        """Execute on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch_idx, batch, loss, outputs)
    
    def on_epoch_end(self, trainer, epoch: int, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]]) -> None:
        """Execute on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, train_metrics, val_metrics)
    
    def on_training_end(self, trainer) -> None:
        """Execute on_training_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_training_end(trainer)
    
    def should_stop_training(self) -> bool:
        """Check if any callback indicates training should stop."""
        for callback in self.callbacks:
            if hasattr(callback, 'should_stop') and callback.should_stop:
                return True
        return False