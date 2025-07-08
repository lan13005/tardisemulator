"""Base model interface for all PyTorch models."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models in the pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize base model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        self.logger = logging.getLogger('pytorch_pipeline')
        
        # Store model metadata
        self.input_dim = config.get('input_dim')
        self.output_dim = config.get('output_dim')
        self.model_type = config.get('model_type', 'base')
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': self.model_type,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'config': self.config
        }
    
    def print_model_summary(self) -> None:
        """Print model summary to console."""
        summary = self.get_model_summary()
        
        print(f"\n{'='*50}")
        print(f"Model Summary: {summary['model_type']}")
        print(f"{'='*50}")
        print(f"Input dimension: {summary['input_dim']}")
        print(f"Output dimension: {summary['output_dim']}")
        print(f"Total parameters: {summary['total_parameters']:,}")
        print(f"Trainable parameters: {summary['trainable_parameters']:,}")
        print(f"Model size: {summary['model_size_mb']:.2f} MB")
        print(f"{'='*50}\n")
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = 0
        trainable_params = 0
        
        for param in self.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }
    
    def freeze_layers(self, layer_names: Optional[list] = None) -> None:
        """Freeze model layers.
        
        Args:
            layer_names: List of layer names to freeze. If None, freeze all layers.
        """
        if layer_names is None:
            # Freeze all parameters
            for param in self.parameters():
                param.requires_grad = False
            self.logger.info("Froze all model parameters")
        else:
            # Freeze specific layers
            for name, module in self.named_modules():
                if name in layer_names:
                    for param in module.parameters():
                        param.requires_grad = False
                    self.logger.info(f"Froze layer: {name}")
    
    def unfreeze_layers(self, layer_names: Optional[list] = None) -> None:
        """Unfreeze model layers.
        
        Args:
            layer_names: List of layer names to unfreeze. If None, unfreeze all layers.
        """
        if layer_names is None:
            # Unfreeze all parameters
            for param in self.parameters():
                param.requires_grad = True
            self.logger.info("Unfroze all model parameters")
        else:
            # Unfreeze specific layers
            for name, module in self.named_modules():
                if name in layer_names:
                    for param in module.parameters():
                        param.requires_grad = True
                    self.logger.info(f"Unfroze layer: {name}")
    
    def get_device(self) -> torch.device:
        """Get the device the model is on.
        
        Returns:
            Device object
        """
        return next(self.parameters()).device
    
    def move_to_device(self, device: torch.device) -> 'BaseModel':
        """Move model to specified device.
        
        Args:
            device: Target device
            
        Returns:
            Self for method chaining
        """
        self.to(device)
        self.logger.info(f"Moved model to device: {device}")
        return self
    
    def save_model_config(self, filepath: str) -> None:
        """Save model configuration to file.
        
        Args:
            filepath: Path to save configuration
        """
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        self.logger.info(f"Saved model config to {filepath}")
    
    @classmethod
    def load_model_config(cls, filepath: str) -> Dict[str, Any]:
        """Load model configuration from file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        import json
        
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        return config
    
    def validate_input(self, x: torch.Tensor) -> None:
        """Validate input tensor dimensions.
        
        Args:
            x: Input tensor
            
        Raises:
            ValueError: If input dimensions are invalid
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor, got {x.dim()}D")
        
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.size(1)}")
    
    def init_weights(self) -> None:
        """Initialize model weights. Should be implemented by subclasses."""
        pass 