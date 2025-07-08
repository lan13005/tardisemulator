"""Logging utilities and TensorBoard integration."""

import os
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    experiment_name: Optional[str] = None
) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to save log files
        experiment_name: Name of the experiment for log file naming
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('pytorch_pipeline')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir specified)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        
        if experiment_name:
            log_filename = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        else:
            log_filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        log_filepath = os.path.join(log_dir, log_filename)
        
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_filepath}")
    
    return logger


class TensorBoardLogger:
    """TensorBoard logging wrapper."""
    
    def __init__(self, log_dir: str, experiment_name: Optional[str] = None):
        """Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
            experiment_name: Name of the experiment
        """
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard not available. Install with: pip install tensorboard")
        
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        # Create experiment-specific log directory
        if experiment_name:
            tensorboard_dir = os.path.join(log_dir, "tensorboard", experiment_name)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            tensorboard_dir = os.path.join(log_dir, "tensorboard", f"experiment_{timestamp}")
        
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value.
        
        Args:
            tag: Name of the scalar
            value: Value to log
            step: Step number (usually epoch or iteration)
        """
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, scalars: Dict[str, float], step: int) -> None:
        """Log multiple scalars.
        
        Args:
            tag: Main tag name
            scalars: Dictionary of {scalar_name: value}
            step: Step number
        """
        self.writer.add_scalars(tag, scalars, step)
    
    def log_histogram(self, tag: str, values, step: int) -> None:
        """Log histogram of values.
        
        Args:
            tag: Name of the histogram
            values: Values to create histogram from
            step: Step number
        """
        self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, image, step: int) -> None:
        """Log an image.
        
        Args:
            tag: Name of the image
            image: Image tensor
            step: Step number
        """
        self.writer.add_image(tag, image, step)
    
    def log_graph(self, model, input_to_model) -> None:
        """Log model graph.
        
        Args:
            model: PyTorch model
            input_to_model: Example input tensor
        """
        self.writer.add_graph(model, input_to_model)
    
    def log_hyperparameters(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, float]) -> None:
        """Log hyperparameters and metrics.
        
        Args:
            hparam_dict: Dictionary of hyperparameters
            metric_dict: Dictionary of metrics
        """
        self.writer.add_hparams(hparam_dict, metric_dict)
    
    def close(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()
    
    def flush(self) -> None:
        """Flush the TensorBoard writer."""
        self.writer.flush()


class ExperimentLogger:
    """Combined logger for experiments with both file and TensorBoard logging."""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        log_level: str = "INFO",
        use_tensorboard: bool = True
    ):
        """Initialize experiment logger.
        
        Args:
            log_dir: Directory for all logs
            experiment_name: Name of the experiment
            log_level: Logging level
            use_tensorboard: Whether to use TensorBoard logging
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard
        
        # Setup standard logging
        self.logger = setup_logging(log_level, log_dir, experiment_name)
        
        # Setup TensorBoard logging
        self.tb_logger = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            try:
                self.tb_logger = TensorBoardLogger(log_dir, experiment_name)
                self.logger.info("TensorBoard logging enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize TensorBoard logging: {e}")
        elif use_tensorboard and not TENSORBOARD_AVAILABLE:
            self.logger.warning("TensorBoard requested but not available")
    
    def log_training_step(
        self,
        epoch: int,
        step: int,
        loss: float,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Log training step information.
        
        Args:
            epoch: Current epoch
            step: Current step
            loss: Training loss
            metrics: Additional metrics
        """
        # Log to file
        self.logger.info(f"Epoch {epoch}, Step {step}: Loss = {loss:.6f}")
        if metrics:
            metric_str = ", ".join([f"{k}={v:.6f}" for k, v in metrics.items()])
            self.logger.info(f"Epoch {epoch}, Step {step}: {metric_str}")
        
        # Log to TensorBoard
        if self.tb_logger:
            self.tb_logger.log_scalar("Training/Loss", loss, step)
            if metrics:
                for name, value in metrics.items():
                    self.tb_logger.log_scalar(f"Training/{name}", value, step)
    
    def log_validation_results(
        self,
        epoch: int,
        val_loss: float,
        val_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Log validation results.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            val_metrics: Validation metrics
        """
        # Log to file
        self.logger.info(f"Epoch {epoch} Validation: Loss = {val_loss:.6f}")
        if val_metrics:
            metric_str = ", ".join([f"{k}={v:.6f}" for k, v in val_metrics.items()])
            self.logger.info(f"Epoch {epoch} Validation: {metric_str}")
        
        # Log to TensorBoard
        if self.tb_logger:
            self.tb_logger.log_scalar("Validation/Loss", val_loss, epoch)
            if val_metrics:
                for name, value in val_metrics.items():
                    self.tb_logger.log_scalar(f"Validation/{name}", value, epoch)
    
    def close(self) -> None:
        """Close all loggers."""
        if self.tb_logger:
            self.tb_logger.close() 