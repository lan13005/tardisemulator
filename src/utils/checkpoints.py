"""Checkpoint management utilities."""

import os
import torch
import logging
from typing import Dict, Any, Optional
from pathlib import Path


class CheckpointManager:
    """Manages model checkpoints and training state."""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        """Initialize CheckpointManager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.logger = logging.getLogger('pytorch_pipeline')
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        additional_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save model checkpoint.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state to save
            scheduler: Learning rate scheduler state
            epoch: Current epoch number
            metrics: Training metrics
            is_best: Whether this is the best model so far
            additional_state: Any additional state to save
            
        Returns:
            Path to saved checkpoint file
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'model_config': getattr(model, 'config', None)
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if additional_state:
            checkpoint.update(additional_state)
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model if this is the best
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model: {best_path}")
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load checkpoint on
            
        Returns:
            Dictionary containing loaded checkpoint information
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'model_config': checkpoint.get('model_config', None)
        }
    
    def load_best_model(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """Load the best saved model.
        
        Args:
            model: Model to load state into
            device: Device to load checkpoint on
            
        Returns:
            Dictionary containing loaded checkpoint information
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        return self.load_checkpoint(str(best_path), model, device=device)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        
        if not checkpoint_files:
            return None
        
        # Sort by epoch number
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return str(checkpoint_files[-1])
    
    def list_checkpoints(self) -> list[str]:
        """List all available checkpoint files.
        
        Returns:
            List of checkpoint file paths
        """
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return [str(f) for f in checkpoint_files]
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to maintain max_checkpoints limit."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        
        if len(checkpoint_files) > self.max_checkpoints:
            # Sort by epoch number
            checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
            
            # Remove oldest checkpoints
            files_to_remove = checkpoint_files[:-self.max_checkpoints]
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    self.logger.debug(f"Removed old checkpoint: {file_path}")
                except OSError as e:
                    self.logger.warning(f"Failed to remove checkpoint {file_path}: {e}")
    
    def save_model_only(
        self,
        model: torch.nn.Module,
        filename: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save model state dict only (for inference).
        
        Args:
            model: PyTorch model to save
            filename: Name of the file to save
            additional_info: Additional information to save with model
            
        Returns:
            Path to saved model file
        """
        model_state = {
            'model_state_dict': model.state_dict(),
            'model_config': getattr(model, 'config', None)
        }
        
        if additional_info:
            model_state.update(additional_info)
        
        model_path = self.checkpoint_dir / filename
        torch.save(model_state, model_path)
        self.logger.info(f"Saved model: {model_path}")
        
        return str(model_path)
    
    def load_model_only(
        self,
        model_path: str,
        model: torch.nn.Module,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """Load model state dict only.
        
        Args:
            model_path: Path to model file
            model: Model to load state into
            device: Device to load model on
            
        Returns:
            Dictionary containing additional model information
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model_state = torch.load(model_path, map_location=device)
        model.load_state_dict(model_state['model_state_dict'])
        
        self.logger.info(f"Loaded model from: {model_path}")
        
        return {
            'model_config': model_state.get('model_config', None)
        } 