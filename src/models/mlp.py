"""Multi-Layer Perceptron (MLP) model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
import logging

from .base_model import BaseModel


class MLP(BaseModel):
    """Configurable Multi-Layer Perceptron model."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MLP model.
        
        Args:
            config: Model configuration dictionary containing:
                - input_dim: Input feature dimension
                - output_dim: Output dimension
                - hidden_dims: List of hidden layer dimensions
                - activation: Activation function name ('relu', 'tanh', 'sigmoid', 'leaky_relu', 'gelu', 'swish')
                - dropout_rate: Dropout probability (default: 0.0)
                - use_batch_norm: Whether to use batch normalization (default: False)
                - use_bias: Whether to use bias in linear layers (default: True)
        """
        super().__init__(config)
        
        # Extract configuration
        self.hidden_dims = config.get('hidden_dims', [128, 64])
        self.activation_name = config.get('activation', 'relu')
        self.dropout_rate = config.get('dropout_rate', 0.0)
        self.use_batch_norm = config.get('use_batch_norm', False)
        self.use_bias = config.get('use_bias', True)
        
        # Validate configuration
        self._validate_config()
        
        # Build the network
        self.layers = self._build_network()
        
        # Initialize weights
        self.init_weights()
        
        self.logger.info(f"Created MLP with architecture: {self._get_architecture_string()}")
    
    def _validate_config(self) -> None:
        """Validate model configuration."""
        if self.input_dim is None or self.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        
        if self.output_dim is None or self.output_dim <= 0:
            raise ValueError("output_dim must be a positive integer")
        
        if not isinstance(self.hidden_dims, list) or len(self.hidden_dims) == 0:
            raise ValueError("hidden_dims must be a non-empty list")
        
        if any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("All hidden dimensions must be positive")
        
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate must be in range [0, 1)")
        
        valid_activations = ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'gelu', 'swish']
        if self.activation_name not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}")
    
    def _get_activation(self) -> nn.Module:
        """Get activation function module.
        
        Returns:
            Activation function module
        """
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'gelu': nn.GELU(),
            'swish': nn.SiLU()  # SiLU is the same as Swish
        }
        return activation_map[self.activation_name]
    
    def _build_network(self) -> nn.ModuleList:
        """Build the neural network layers.
        
        Returns:
            ModuleList containing all network layers
        """
        layers = nn.ModuleList()
        
        # Create list of all layer dimensions
        all_dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        
        # Build hidden layers
        for i in range(len(all_dims) - 1):
            in_dim = all_dims[i]
            out_dim = all_dims[i + 1]
            is_output_layer = (i == len(all_dims) - 2)
            
            # Linear layer
            linear = nn.Linear(in_dim, out_dim, bias=self.use_bias)
            layers.append(linear)
            
            # Don't add activation, batch norm, or dropout to output layer
            if not is_output_layer:
                # Batch normalization
                #   before activation, ReLU would half-rectify it which defeats purpose of zero-centering
                #   this ensures ReLU would activate half the time maximizing gradient flow
                if self.use_batch_norm:
                    layers.append(nn.BatchNorm1d(out_dim))
                
                # Activation function
                layers.append(self._get_activation())
                
                # Dropout
                #    must be after batch norm else BatchNorm stats will change and attempt to compensate for dropout
                #    would introduce more noise
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout(self.dropout_rate))
        
        return layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Validate input
        self.validate_input(x)
        
        # Forward pass through all layers
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def forward_with_intermediate(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning intermediate layer outputs.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with layer outputs
        """
        self.validate_input(x)
        
        outputs = {'input': x}
        layer_idx = 0
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Save outputs after each complete block (linear + optional batch norm + activation + dropout)
            if isinstance(layer, nn.Linear):
                layer_idx += 1
                outputs[f'linear_{layer_idx}'] = x
            elif isinstance(layer, nn.BatchNorm1d):
                outputs[f'batch_norm_{layer_idx}'] = x
            elif isinstance(layer, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU, nn.GELU, nn.SiLU)):
                outputs[f'activation_{layer_idx}'] = x
            elif isinstance(layer, nn.Dropout):
                outputs[f'dropout_{layer_idx}'] = x
        
        outputs['output'] = x
        return outputs
    
    def get_layer_outputs(self, x: torch.Tensor, layer_indices: List[int]) -> List[torch.Tensor]:
        """Get outputs from specific layers.
        
        Args:
            x: Input tensor
            layer_indices: List of layer indices to return outputs from
            
        Returns:
            List of tensors from specified layers
        """
        self.validate_input(x)
        
        outputs = []
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in layer_indices:
                outputs.append(x.clone())
        
        return outputs
    
    def init_weights(self) -> None:
        """Initialize model weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for linear layers
                if self.activation_name in ['relu', 'leaky_relu']:
                    # Kaiming initialization for ReLU-like activations
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                else:
                    # Xavier initialization for other activations
                    nn.init.xavier_normal_(module.weight)
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            elif isinstance(module, nn.BatchNorm1d):
                # Initialize batch norm parameters
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def _get_architecture_string(self) -> str:
        """Get string representation of model architecture.
        
        Returns:
            Architecture string
        """
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        arch_str = " -> ".join(map(str, dims))
        
        details = []
        if self.use_batch_norm:
            details.append("BatchNorm")
        if self.dropout_rate > 0:
            details.append(f"Dropout({self.dropout_rate})")
        
        if details:
            arch_str += f" [{', '.join(details)}]"
        
        return arch_str
    
    def get_layer_info(self) -> List[Dict[str, Any]]:
        """Get information about each layer.
        
        Returns:
            List of dictionaries with layer information
        """
        layer_info = []
        
        for i, layer in enumerate(self.layers):
            info = {
                'index': i,
                'type': type(layer).__name__,
                'parameters': sum(p.numel() for p in layer.parameters())
            }
            
            if isinstance(layer, nn.Linear):
                info['input_features'] = layer.in_features
                info['output_features'] = layer.out_features
                info['bias'] = layer.bias is not None
            elif isinstance(layer, nn.BatchNorm1d):
                info['num_features'] = layer.num_features
            elif isinstance(layer, nn.Dropout):
                info['dropout_rate'] = layer.p
            
            layer_info.append(info)
        
        return layer_info
    
    def prune_neurons(self, layer_idx: int, neuron_indices: List[int]) -> None:
        """Prune specific neurons from a layer (for model compression).
        
        Args:
            layer_idx: Index of the layer to prune
            neuron_indices: List of neuron indices to remove
        """
        # This is a simplified implementation
        # In practice, you'd need to modify the weights of this layer and the next layer
        self.logger.warning("Neuron pruning is not fully implemented - this is a placeholder")
    
    def add_layer(self, position: int, layer_dim: int) -> None:
        """Add a new hidden layer at the specified position.
        
        Args:
            position: Position to insert the layer (0-indexed)
            layer_dim: Dimension of the new layer
        """
        # This would require rebuilding the entire network
        self.logger.warning("Dynamic layer addition is not implemented - rebuild the model instead")
    
    def get_gradient_info(self) -> Dict[str, Any]:
        """Get information about gradients (useful for debugging).
        
        Returns:
            Dictionary with gradient statistics
        """
        grad_info = {}
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                grad_info[name] = {
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'std': grad_std,
                    'max': param.grad.max().item(),
                    'min': param.grad.min().item()
                }
            else:
                grad_info[name] = None
        
        return grad_info 