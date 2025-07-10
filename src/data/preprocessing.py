"""Data preprocessing utilities."""

import torch
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging


class DataPreprocessor:
    """Data preprocessing utilities for normalization and standardization."""
    
    def __init__(self, method: str = 'standard', log_scaling: bool = False, output_log_scaling: bool = False):
        """Initialize DataPreprocessor.
        
        Args:
            method: Preprocessing method ('standard', 'minmax', 'robust', 'none')
            log_scaling: Whether to apply log10 scaling to columns with index >= 3
            output_log_scaling: Whether to apply log10 scaling to all output columns (spectra)
        """
        self.method = method
        self.log_scaling = log_scaling
        self.output_log_scaling = output_log_scaling
        self.input_scaler = None
        self.output_scaler = None
        self.is_fitted = False
        self.logger = logging.getLogger('pytorch_pipeline')
        
        # Initialize scalers based on method
        if method == 'standard':
            self.input_scaler = StandardScaler()
            self.output_scaler = StandardScaler()
        elif method == 'minmax':
            self.input_scaler = MinMaxScaler()
            self.output_scaler = MinMaxScaler()
        elif method == 'robust':
            self.input_scaler = RobustScaler()
            self.output_scaler = RobustScaler()
        elif method == 'none':
            self.input_scaler = None
            self.output_scaler = None
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
    
    def _apply_log_scaling(self, data: np.ndarray) -> np.ndarray:
        """Apply log10 scaling to columns with index >= 3.
        
        Args:
            data: Input data array
            
        Returns:
            Data with log scaling applied to appropriate columns
        """
        if not self.log_scaling or data.shape[1] <= 3:
            return data
        data_copy = data.copy()
        data_copy[:, 3:] = np.log10(data_copy[:, 3:])
        return data_copy
    
    def _inverse_log_scaling(self, data: np.ndarray) -> np.ndarray:
        """Inverse log10 scaling for columns with index >= 3.
        
        Args:
            data: Input data array with log scaling applied
            
        Returns:
            Data with log scaling reversed
        """
        if not self.log_scaling or data.shape[1] <= 3:
            return data
        data_copy = data.copy()
        data_copy[:, 3:] = np.power(10, data_copy[:, 3:])
        return data_copy
    
    def _apply_output_log_scaling(self, data: np.ndarray) -> np.ndarray:
        """Apply log10 scaling to all output columns (spectra).
        
        Args:
            data: Output data array (spectra)
            
        Returns:
            Data with log scaling applied to all columns
        """
        if not self.output_log_scaling:
            return data
        data_copy = data.copy()
        data_copy = np.log10(data_copy)
        return data_copy
    
    def _inverse_output_log_scaling(self, data: np.ndarray) -> np.ndarray:
        """Inverse log10 scaling for all output columns (spectra).
        
        Args:
            data: Output data array with log scaling applied
            
        Returns:
            Data with log scaling reversed
        """
        if not self.output_log_scaling:
            return data
        data_copy = data.copy()
        data_copy = np.power(10, data_copy)
        return data_copy
    
    @property
    def scaler(self):
        """Get the output scaler for inverse transformation in diagnostic plotting.
        
        Returns:
            Output scaler object or None if not fitted
        """
        if self.is_fitted and self.output_scaler is not None:
            return self.output_scaler
        return None
    
    def fit(
        self,
        input_data: torch.Tensor,
        output_data: torch.Tensor
    ) -> 'DataPreprocessor':
        """Fit preprocessing scalers to data.
        
        Args:
            input_data: Input features tensor
            output_data: Target values tensor
            
        Returns:
            Self for method chaining
        """
        if self.method == 'none':
            self.is_fitted = True
            return self
        
        # Convert tensors to numpy for sklearn
        input_np = input_data.detach().cpu().numpy() if isinstance(input_data, torch.Tensor) else input_data
        output_np = output_data.detach().cpu().numpy() if isinstance(output_data, torch.Tensor) else output_data
        
        # Apply log scaling to input data before fitting scaler
        if self.log_scaling:
            input_np = self._apply_log_scaling(input_np)
            self.logger.info(f"Applied log scaling to input columns with index >= 3")
        
        # Apply log scaling to output data before fitting scaler
        if self.output_log_scaling:
            output_np = self._apply_output_log_scaling(output_np)
            self.logger.info(f"Applied log scaling to all output columns (spectra)")
        
        # Fit scalers
        if self.input_scaler is not None:
            self.input_scaler.fit(input_np)
            self.logger.info(f"Fitted input scaler ({self.method}) to data shape {input_np.shape}")
        
        if self.output_scaler is not None:
            self.output_scaler.fit(output_np)
            self.logger.info(f"Fitted output scaler ({self.method}) to data shape {output_np.shape}")
        
        self.is_fitted = True
        return self
    
    def transform_input(self, input_data: torch.Tensor) -> torch.Tensor:
        """Transform input data using fitted scaler.
        
        Args:
            input_data: Input features tensor
            
        Returns:
            Transformed input tensor
        """
        # Handle empty tensors gracefully (even if not fitted)
        if isinstance(input_data, torch.Tensor) and input_data.shape[0] == 0:
            return input_data  # Return empty tensor as-is
        elif not isinstance(input_data, torch.Tensor) and len(input_data) == 0:
            return input_data  # Return empty array as-is
        
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        # Convert to numpy
        input_np = input_data.detach().cpu().numpy() if isinstance(input_data, torch.Tensor) else input_data
        
        # Apply log scaling first
        if self.log_scaling:
            input_np = self._apply_log_scaling(input_np)
        
        # Then apply main scaler
        if self.input_scaler is not None:
            transformed_np = self.input_scaler.transform(input_np)
        else:
            transformed_np = input_np
        
        if isinstance(input_data, torch.Tensor):
            return torch.tensor(transformed_np, dtype=input_data.dtype, device=input_data.device)
        else:
            return transformed_np
    
    def transform_output(self, output_data: torch.Tensor) -> torch.Tensor:
        """Transform output data using fitted scaler.
        
        Args:
            output_data: Target values tensor
            
        Returns:
            Transformed output tensor
        """
        # Handle empty tensors gracefully (even if not fitted)
        if isinstance(output_data, torch.Tensor) and output_data.shape[0] == 0:
            return output_data  # Return empty tensor as-is
        elif not isinstance(output_data, torch.Tensor) and len(output_data) == 0:
            return output_data  # Return empty array as-is
        
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        # Convert to numpy
        output_np = output_data.detach().cpu().numpy() if isinstance(output_data, torch.Tensor) else output_data
        
        # Apply log scaling first
        if self.output_log_scaling:
            output_np = self._apply_output_log_scaling(output_np)
        
        # Then apply main scaler
        if self.output_scaler is not None:
            transformed_np = self.output_scaler.transform(output_np)
        else:
            transformed_np = output_np
        
        if isinstance(output_data, torch.Tensor):
            return torch.tensor(transformed_np, dtype=output_data.dtype, device=output_data.device)
        else:
            return transformed_np
    
    def inverse_transform_input(self, input_data: torch.Tensor) -> torch.Tensor:
        """Inverse transform input data.
        
        Args:
            input_data: Transformed input features tensor
            
        Returns:
            Original scale input tensor
        """
        # Handle empty tensors gracefully (even if not fitted)
        if isinstance(input_data, torch.Tensor) and input_data.shape[0] == 0:
            return input_data  # Return empty tensor as-is
        elif not isinstance(input_data, torch.Tensor) and len(input_data) == 0:
            return input_data  # Return empty array as-is
        
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        # Convert to numpy
        input_np = input_data.detach().cpu().numpy() if isinstance(input_data, torch.Tensor) else input_data
        
        # First inverse transform main scaler
        if self.input_scaler is not None:
            original_np = self.input_scaler.inverse_transform(input_np)
        else:
            original_np = input_np
        
        # Then inverse transform log scaling
        if self.log_scaling:
            original_np = self._inverse_log_scaling(original_np)
        
        if isinstance(input_data, torch.Tensor):
            return torch.tensor(original_np, dtype=input_data.dtype, device=input_data.device)
        else:
            return original_np
    
    def inverse_transform_output(self, output_data: torch.Tensor) -> torch.Tensor:
        """Inverse transform output data.
        
        Args:
            output_data: Transformed target values tensor
            
        Returns:
            Original scale output tensor
        """
        # Handle empty tensors gracefully (even if not fitted)
        if isinstance(output_data, torch.Tensor) and output_data.shape[0] == 0:
            return output_data  # Return empty tensor as-is
        elif not isinstance(output_data, torch.Tensor) and len(output_data) == 0:
            return output_data  # Return empty array as-is
        
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        # Convert to numpy
        output_np = output_data.detach().cpu().numpy() if isinstance(output_data, torch.Tensor) else output_data
        
        # First inverse transform main scaler
        if self.output_scaler is not None:
            original_np = self.output_scaler.inverse_transform(output_np)
        else:
            original_np = output_np
        
        # Then inverse transform log scaling
        if self.output_log_scaling:
            original_np = self._inverse_output_log_scaling(original_np)
        
        if isinstance(output_data, torch.Tensor):
            return torch.tensor(original_np, dtype=output_data.dtype, device=output_data.device)
        else:
            return original_np
    
    def fit_transform(
        self,
        input_data: torch.Tensor,
        output_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fit preprocessor and transform data in one step.
        
        Args:
            input_data: Input features tensor
            output_data: Target values tensor
            
        Returns:
            Tuple of (transformed_input, transformed_output)
        """
        self.fit(input_data, output_data)
        transformed_input = self.transform_input(input_data)
        transformed_output = self.transform_output(output_data)
        return transformed_input, transformed_output
    
    def get_scaler_info(self) -> Dict[str, Any]:
        """Get information about fitted scalers.
        
        Returns:
            Dictionary with scaler information
        """
        if not self.is_fitted:
            return {'method': self.method, 'log_scaling': self.log_scaling, 'output_log_scaling': self.output_log_scaling, 'fitted': False}
        
        info = {
            'method': self.method,
            'log_scaling': self.log_scaling,
            'output_log_scaling': self.output_log_scaling,
            'fitted': True
        }
        
        if self.input_scaler is not None:
            if hasattr(self.input_scaler, 'mean_'):
                info['input_mean'] = self.input_scaler.mean_
                info['input_std'] = self.input_scaler.scale_
            elif hasattr(self.input_scaler, 'data_min_'):
                info['input_min'] = self.input_scaler.data_min_
                info['input_max'] = self.input_scaler.data_max_
            elif hasattr(self.input_scaler, 'center_'):
                info['input_center'] = self.input_scaler.center_
                info['input_scale'] = self.input_scaler.scale_
        
        if self.output_scaler is not None:
            if hasattr(self.output_scaler, 'mean_'):
                info['output_mean'] = self.output_scaler.mean_
                info['output_std'] = self.output_scaler.scale_
            elif hasattr(self.output_scaler, 'data_min_'):
                info['output_min'] = self.output_scaler.data_min_
                info['output_max'] = self.output_scaler.data_max_
            elif hasattr(self.output_scaler, 'center_'):
                info['output_center'] = self.output_scaler.center_
                info['output_scale'] = self.output_scaler.scale_
        
        return info
    
    def save_scalers(self, filepath: str) -> None:
        """Save fitted scalers to file.
        
        Args:
            filepath: Path to save scalers
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        import pickle
        
        scaler_data = {
            'method': self.method,
            'log_scaling': self.log_scaling,
            'output_log_scaling': self.output_log_scaling,
            'input_scaler': self.input_scaler,
            'output_scaler': self.output_scaler,
            'is_fitted': self.is_fitted,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(scaler_data, f)
        
        self.logger.info(f"Saved scalers to {filepath}")
    
    def load_scalers(self, filepath: str) -> None:
        """Load fitted scalers from file.
        
        Args:
            filepath: Path to load scalers from
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            scaler_data = pickle.load(f)
        
        self.method = scaler_data['method']
        self.log_scaling = scaler_data.get('log_scaling', False)  # Backward compatibility
        self.output_log_scaling = scaler_data.get('output_log_scaling', False)  # Backward compatibility
        self.input_scaler = scaler_data['input_scaler']
        self.output_scaler = scaler_data['output_scaler']
        self.is_fitted = scaler_data['is_fitted']
        
        self.logger.info(f"Loaded scalers from {filepath}")


class DataAugmenter:
    """Data augmentation utilities."""
    
    def __init__(self, noise_std: float = 0.01, dropout_prob: float = 0.1):
        """Initialize DataAugmenter.
        
        Args:
            noise_std: Standard deviation for Gaussian noise
            dropout_prob: Probability for feature dropout
        """
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.logger = logging.getLogger('pytorch_pipeline')
    
    def add_gaussian_noise(self, data: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to data.
        
        Args:
            data: Input data tensor
            
        Returns:
            Data with added noise
        """
        noise = torch.randn_like(data) * self.noise_std
        return data + noise
    
    def feature_dropout(self, data: torch.Tensor) -> torch.Tensor:
        """Apply random feature dropout.
        
        Args:
            data: Input data tensor
            
        Returns:
            Data with random features set to zero
        """
        mask = torch.rand_like(data) > self.dropout_prob
        return data * mask.float()
    
    def augment_batch(
        self,
        input_batch: torch.Tensor,
        output_batch: torch.Tensor,
        apply_noise: bool = True,
        apply_dropout: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation to a batch of data.
        
        Args:
            input_batch: Input features batch
            output_batch: Target values batch
            apply_noise: Whether to apply Gaussian noise
            apply_dropout: Whether to apply feature dropout
            
        Returns:
            Tuple of (augmented_input, original_output)
        """
        augmented_input = input_batch.clone()
        
        if apply_noise:
            augmented_input = self.add_gaussian_noise(augmented_input)
        
        if apply_dropout:
            augmented_input = self.feature_dropout(augmented_input)
        
        return augmented_input, output_batch 