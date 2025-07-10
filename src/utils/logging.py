"""Logging utilities and MLflow integration."""

import os
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


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


class MLflowLogger:
    """MLflow logging wrapper."""
    
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        """Initialize MLflow logger.
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI (optional, defaults to local file system)
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not available. Install with: pip install mlflow")
        
        self.experiment_name = experiment_name
        
        # Set tracking URI - use "mlruns" as the backend for all experiments
        # This ensures all MLflow data goes to the centralized mlruns/ directory
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Default to local file system backend
            mlflow.set_tracking_uri("mlruns")
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        # Start run
        self.run = mlflow.start_run()
        self.logger = logging.getLogger('pytorch_pipeline')
        self.logger.info(f"MLflow run started: {self.run.info.run_id}")
        self.logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        self.logger.info(f"MLflow experiment: {experiment_name}")
    
    def log_scalar(self, key: str, value: float, step: int) -> None:
        """Log a scalar value.
        
        Args:
            key: Name of the scalar
            value: Value to log
            step: Step number (usually epoch or iteration)
        """
        mlflow.log_metric(key, value, step=step)
    
    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        """Log multiple scalars.
        
        Args:
            metrics: Dictionary of {metric_name: value}
            step: Step number
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_histogram(self, key: str, values, step: int) -> None:
        """Log histogram of values.
        
        Args:
            key: Name of the histogram
            values: Values to create histogram from
            step: Step number
        """
        # MLflow doesn't have direct histogram logging, but we can log as artifact
        import numpy as np
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.hist(values, bins=50, alpha=0.7)
        ax.set_title(f'{key} (step {step})')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        
        # Save and log as artifact
        hist_path = f"histograms/{key}_step_{step}.png"
        os.makedirs(os.path.dirname(hist_path), exist_ok=True)
        plt.savefig(hist_path)
        mlflow.log_artifact(hist_path)
        plt.close()
    
    def log_image(self, key: str, image_path: str, step: int) -> None:
        """Log an image.
        
        Args:
            key: Name of the image
            image_path: Path to the image file
            step: Step number
        """
        mlflow.log_artifact(image_path, artifact_path=f"images/{key}_step_{step}")
    
    def log_model(self, model, artifact_path: str = "model") -> None:
        """Log PyTorch model.
        
        Args:
            model: PyTorch model
            artifact_path: Path where to save the model
        """
        mlflow.pytorch.log_model(model, artifact_path)
    
    def log_hyperparameters(self, hparam_dict: Dict[str, Any]) -> None:
        """Log hyperparameters.
        
        Args:
            hparam_dict: Dictionary of hyperparameters
        """
        mlflow.log_params(hparam_dict)
    
    def log_config(self, config: Dict[str, Any], config_name: str = "config.yaml") -> None:
        """Log configuration file.
        
        Args:
            config: Configuration dictionary
            config_name: Name of the config file
        """
        # Save config as YAML file
        config_path = f"configs/{config_name}"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        mlflow.log_artifact(config_path)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact.
        
        Args:
            local_path: Path to the local file/directory
            artifact_path: Path within the artifact store
        """
        mlflow.log_artifact(local_path, artifact_path)
    
    def close(self) -> None:
        """End the MLflow run."""
        mlflow.end_run()
    
    def get_run_id(self) -> str:
        """Get the current run ID."""
        return self.run.info.run_id


class ExperimentLogger:
    """Combined logger for experiments with both file and MLflow logging."""
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: Optional[str] = None,
        log_level: str = "INFO",
        use_mlflow: bool = True,
        tracking_uri: Optional[str] = None
    ):
        """Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for file logs
            log_level: Logging level
            use_mlflow: Whether to use MLflow logging
            tracking_uri: MLflow tracking URI
        """
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow
        
        # Setup standard logging
        self.logger = setup_logging(log_level, log_dir, experiment_name)
        
        # Setup MLflow logging
        self.mlflow_logger = None
        if use_mlflow and MLFLOW_AVAILABLE:
            try:
                self.mlflow_logger = MLflowLogger(experiment_name, tracking_uri)
                self.logger.info("MLflow logging enabled")
                self.logger.info(f"MLflow tracking backend: {tracking_uri or 'mlruns'}")
                self.logger.info(f"Local experiment outputs: {log_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize MLflow logging: {e}")
        elif use_mlflow and not MLFLOW_AVAILABLE:
            self.logger.warning("MLflow requested but not available")
    
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
        
        # Log to MLflow
        if self.mlflow_logger:
            self.mlflow_logger.log_scalar("Training/Loss", loss, step)
            if metrics:
                for name, value in metrics.items():
                    self.mlflow_logger.log_scalar(f"Training/{name}", value, step)
    
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
        
        # Log to MLflow
        if self.mlflow_logger:
            self.mlflow_logger.log_scalar("Validation/Loss", val_loss, epoch)
            if val_metrics:
                for name, value in val_metrics.items():
                    self.mlflow_logger.log_scalar(f"Validation/{name}", value, epoch)
    
    def log_model(self, model, artifact_path: str = "model") -> None:
        """Log PyTorch model.
        
        Args:
            model: PyTorch model
            artifact_path: Path where to save the model
        """
        if self.mlflow_logger:
            self.mlflow_logger.log_model(model, artifact_path)
    
    def log_config(self, config: Dict[str, Any], config_name: str = "config.yaml") -> None:
        """Log configuration file.
        
        Args:
            config: Configuration dictionary
            config_name: Name of the config file
        """
        if self.mlflow_logger:
            self.mlflow_logger.log_config(config, config_name)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact.
        
        Args:
            local_path: Path to the local file/directory
            artifact_path: Path within the artifact store
        """
        if self.mlflow_logger:
            self.mlflow_logger.log_artifact(local_path, artifact_path)
    
    def get_run_id(self) -> Optional[str]:
        """Get the current MLflow run ID."""
        if self.mlflow_logger:
            return self.mlflow_logger.get_run_id()
        return None
    
    def close(self) -> None:
        """Close all loggers."""
        if self.mlflow_logger:
            self.mlflow_logger.close() 