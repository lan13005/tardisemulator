"""Model inference and prediction utilities."""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import logging

from ..models.base_model import BaseModel
from ..models.mlp import MLP
from ..data.preprocessing import DataPreprocessor
from ..utils.checkpoints import CheckpointManager


class Predictor:
    """Model inference and prediction class."""
    
    def __init__(
        self,
        model: Optional[BaseModel] = None,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        preprocessor: Optional[DataPreprocessor] = None
    ):
        """Initialize Predictor.
        
        Args:
            model: Pre-loaded model instance
            model_path: Path to saved model file
            config_path: Path to model configuration file
            device: Device to run inference on
            preprocessor: Data preprocessor instance
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger('pytorch_pipeline')
        
        # Initialize model
        if model is not None:
            self.model = model
            self.model.to(self.device)
            self.model.eval()
        elif model_path is not None:
            self.model = self._load_model_from_path(model_path, config_path)
        else:
            self.model = None
            self.logger.warning("No model provided. Load a model before making predictions.")
        
        # Initialize preprocessor
        self.preprocessor = preprocessor
        
        self.logger.info(f"Initialized predictor on device: {self.device}")
    
    def _load_model_from_path(self, model_path: str, config_path: Optional[str] = None) -> BaseModel:
        """Load model from file path.
        
        Args:
            model_path: Path to model file
            config_path: Path to configuration file
            
        Returns:
            Loaded model instance
        """
        # Load model state
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model configuration
        if 'model_config' in checkpoint and checkpoint['model_config'] is not None:
            model_config = checkpoint['model_config']
        elif config_path is not None:
            model_config = BaseModel.load_model_config(config_path)
        else:
            raise ValueError("Model configuration not found. Provide config_path or ensure model was saved with config.")
        
        # Create model instance
        model_type = model_config.get('model_type', 'mlp')
        if model_type == 'mlp':
            model = MLP(model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        self.logger.info(f"Loaded {model_type} model from {model_path}")
        
        return model
    
    def load_model(self, model_path: str, config_path: Optional[str] = None) -> None:
        """Load model from file path.
        
        Args:
            model_path: Path to model file
            config_path: Path to configuration file
        """
        self.model = self._load_model_from_path(model_path, config_path)
    
    def load_preprocessor(self, preprocessor_path: str) -> None:
        """Load preprocessor from file.
        
        Args:
            preprocessor_path: Path to saved preprocessor
        """
        self.preprocessor = DataPreprocessor()
        self.preprocessor.load_scalers(preprocessor_path)
        self.logger.info(f"Loaded preprocessor from {preprocessor_path}")
    
    def predict(
        self,
        inputs: Union[torch.Tensor, np.ndarray, pd.DataFrame, List],
        apply_preprocessing: bool = True,
        return_probabilities: bool = False,
        batch_size: Optional[int] = None
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """Make predictions on input data.
        
        Args:
            inputs: Input data (various formats supported)
            apply_preprocessing: Whether to apply preprocessing
            return_probabilities: Whether to return class probabilities (for classification)
            batch_size: Batch size for large inputs (if None, process all at once)
            
        Returns:
            Dictionary containing predictions and optionally probabilities
        """
        if self.model is None:
            raise ValueError("No model loaded. Load a model before making predictions.")
        
        # Convert inputs to tensor
        input_tensor = self._prepare_inputs(inputs, apply_preprocessing)
        
        # Make predictions
        if batch_size is not None and len(input_tensor) > batch_size:
            predictions, probabilities = self._predict_in_batches(
                input_tensor, batch_size, return_probabilities
            )
        else:
            predictions, probabilities = self._predict_single_batch(
                input_tensor, return_probabilities
            )
        
        # Prepare output
        result = {'predictions': predictions}
        if return_probabilities and probabilities is not None:
            result['probabilities'] = probabilities
        
        return result
    
    def _prepare_inputs(
        self,
        inputs: Union[torch.Tensor, np.ndarray, pd.DataFrame, List],
        apply_preprocessing: bool
    ) -> torch.Tensor:
        """Prepare inputs for model inference.
        
        Args:
            inputs: Input data
            apply_preprocessing: Whether to apply preprocessing
            
        Returns:
            Preprocessed input tensor
        """
        # Convert to tensor
        if isinstance(inputs, torch.Tensor):
            input_tensor = inputs
        elif isinstance(inputs, np.ndarray):
            input_tensor = torch.tensor(inputs, dtype=torch.float32)
        elif isinstance(inputs, pd.DataFrame):
            input_tensor = torch.tensor(inputs.values, dtype=torch.float32)
        elif isinstance(inputs, list):
            input_tensor = torch.tensor(inputs, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")
        
        # Ensure 2D tensor
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        elif input_tensor.dim() > 2:
            raise ValueError(f"Input tensor must be 1D or 2D, got {input_tensor.dim()}D")
        
        # Apply preprocessing if requested
        if apply_preprocessing and self.preprocessor is not None:
            input_tensor = self.preprocessor.transform_input(input_tensor)
        
        # Move to device
        input_tensor = input_tensor.to(self.device)
        
        return input_tensor
    
    def _predict_single_batch(
        self,
        input_tensor: torch.Tensor,
        return_probabilities: bool
    ) -> tuple:
        """Make predictions on a single batch.
        
        Args:
            input_tensor: Input tensor
            return_probabilities: Whether to return probabilities
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            # Process outputs based on task type
            if hasattr(self.model, 'config'):
                task_type = self.model.config.get('task_type', 'regression')
            else:
                # Infer task type from output shape
                if outputs.shape[1] == 1:
                    task_type = 'regression'
                else:
                    task_type = 'classification'
            
            if task_type == 'regression':
                predictions = outputs
                probabilities = None
            elif task_type == 'binary_classification':
                probabilities = torch.sigmoid(outputs) if return_probabilities else None
                predictions = (torch.sigmoid(outputs) > 0.5).float()
            else:  # multiclass classification
                probabilities = torch.softmax(outputs, dim=1) if return_probabilities else None
                predictions = torch.argmax(outputs, dim=1)
        
        return predictions, probabilities
    
    def _predict_in_batches(
        self,
        input_tensor: torch.Tensor,
        batch_size: int,
        return_probabilities: bool
    ) -> tuple:
        """Make predictions in batches for large inputs.
        
        Args:
            input_tensor: Input tensor
            batch_size: Batch size
            return_probabilities: Whether to return probabilities
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        all_predictions = []
        all_probabilities = [] if return_probabilities else None
        
        num_samples = input_tensor.shape[0]
        
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_input = input_tensor[i:end_idx]
            
            batch_predictions, batch_probabilities = self._predict_single_batch(
                batch_input, return_probabilities
            )
            
            all_predictions.append(batch_predictions)
            if return_probabilities and batch_probabilities is not None:
                all_probabilities.append(batch_probabilities)
        
        # Concatenate results
        predictions = torch.cat(all_predictions, dim=0)
        probabilities = torch.cat(all_probabilities, dim=0) if all_probabilities else None
        
        return predictions, probabilities
    
    def predict_single(
        self,
        input_sample: Union[torch.Tensor, np.ndarray, List],
        apply_preprocessing: bool = True,
        return_probabilities: bool = False
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Make prediction on a single sample.
        
        Args:
            input_sample: Single input sample
            apply_preprocessing: Whether to apply preprocessing
            return_probabilities: Whether to return probabilities
            
        Returns:
            Dictionary with prediction results
        """
        result = self.predict(
            input_sample, apply_preprocessing, return_probabilities, batch_size=None
        )
        
        # Convert to scalars/arrays for single sample
        predictions = result['predictions'].cpu().numpy()
        if predictions.shape[0] == 1:
            if predictions.shape[1] == 1:
                result['predictions'] = float(predictions[0, 0])
            else:
                result['predictions'] = predictions[0]
        
        if 'probabilities' in result and result['probabilities'] is not None:
            probabilities = result['probabilities'].cpu().numpy()
            if probabilities.shape[0] == 1:
                result['probabilities'] = probabilities[0]
        
        return result
    
    def predict_dataframe(
        self,
        df: pd.DataFrame,
        apply_preprocessing: bool = True,
        return_probabilities: bool = False,
        batch_size: Optional[int] = None
    ) -> pd.DataFrame:
        """Make predictions on DataFrame and return results as DataFrame.
        
        Args:
            df: Input DataFrame
            apply_preprocessing: Whether to apply preprocessing
            return_probabilities: Whether to return probabilities
            batch_size: Batch size for processing
            
        Returns:
            DataFrame with predictions
        """
        result = self.predict(df, apply_preprocessing, return_probabilities, batch_size)
        
        # Create output DataFrame
        predictions = result['predictions'].cpu().numpy()
        
        if predictions.shape[1] == 1:
            # Regression or binary classification
            output_df = pd.DataFrame({'prediction': predictions.flatten()})
        else:
            # Multi-class classification
            pred_cols = [f'prediction_class_{i}' for i in range(predictions.shape[1])]
            output_df = pd.DataFrame(predictions, columns=pred_cols)
        
        # Add probabilities if requested
        if 'probabilities' in result and result['probabilities'] is not None:
            probabilities = result['probabilities'].cpu().numpy()
            if probabilities.shape[1] == 1:
                output_df['probability'] = probabilities.flatten()
            else:
                prob_cols = [f'probability_class_{i}' for i in range(probabilities.shape[1])]
                prob_df = pd.DataFrame(probabilities, columns=prob_cols)
                output_df = pd.concat([output_df, prob_df], axis=1)
        
        return output_df
    
    def get_feature_importance(
        self,
        input_sample: Union[torch.Tensor, np.ndarray],
        method: str = 'gradient'
    ) -> np.ndarray:
        """Get feature importance for a given input.
        
        Args:
            input_sample: Input sample
            method: Method for computing importance ('gradient', 'integrated_gradients')
            
        Returns:
            Feature importance scores
        """
        if self.model is None:
            raise ValueError("No model loaded.")
        
        # Prepare input
        if not isinstance(input_sample, torch.Tensor):
            input_sample = torch.tensor(input_sample, dtype=torch.float32)
        
        if input_sample.dim() == 1:
            input_sample = input_sample.unsqueeze(0)
        
        input_sample = input_sample.to(self.device)
        input_sample.requires_grad_(True)
        
        if method == 'gradient':
            return self._gradient_importance(input_sample)
        elif method == 'integrated_gradients':
            return self._integrated_gradients_importance(input_sample)
        else:
            raise ValueError(f"Unknown importance method: {method}")
    
    def _gradient_importance(self, input_sample: torch.Tensor) -> np.ndarray:
        """Calculate feature importance using gradients.
        
        Args:
            input_sample: Input sample tensor
            
        Returns:
            Feature importance scores
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_sample)
        
        # For regression, use the output directly
        # For classification, use the max probability class
        if output.shape[1] == 1:
            target_output = output[0, 0]
        else:
            target_output = output[0, torch.argmax(output[0])]
        
        # Backward pass
        self.model.zero_grad()
        target_output.backward()
        
        # Get gradients
        gradients = input_sample.grad.abs().cpu().numpy()
        
        return gradients[0]  # Return for single sample
    
    def _integrated_gradients_importance(
        self,
        input_sample: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> np.ndarray:
        """Calculate feature importance using integrated gradients.
        
        Args:
            input_sample: Input sample tensor
            baseline: Baseline input (default: zeros)
            steps: Number of integration steps
            
        Returns:
            Feature importance scores
        """
        if baseline is None:
            baseline = torch.zeros_like(input_sample)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps, device=self.device)
        interpolated_inputs = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (input_sample - baseline)
            interpolated_inputs.append(interpolated)
        
        # Calculate gradients for each interpolated input
        gradients = []
        
        for interpolated in interpolated_inputs:
            interpolated.requires_grad_(True)
            output = self.model(interpolated)
            
            if output.shape[1] == 1:
                target_output = output[0, 0]
            else:
                target_output = output[0, torch.argmax(output[0])]
            
            self.model.zero_grad()
            target_output.backward()
            
            gradients.append(interpolated.grad.clone())
        
        # Average gradients and multiply by input difference
        avg_gradients = torch.stack(gradients).mean(dim=0)
        integrated_gradients = (input_sample - baseline) * avg_gradients
        
        return integrated_gradients.abs().cpu().numpy()[0]
    
    def save_predictions(
        self,
        predictions: Union[Dict, pd.DataFrame],
        filepath: str,
        format: str = 'csv'
    ) -> None:
        """Save predictions to file.
        
        Args:
            predictions: Predictions to save
            filepath: Output file path
            format: Output format ('csv', 'json', 'pickle')
        """
        if format == 'csv':
            if isinstance(predictions, dict):
                df = pd.DataFrame(predictions)
            else:
                df = predictions
            df.to_csv(filepath, index=False)
        
        elif format == 'json':
            if isinstance(predictions, pd.DataFrame):
                predictions.to_json(filepath, orient='records')
            else:
                import json
                # Convert tensors to lists for JSON serialization
                json_data = {}
                for key, value in predictions.items():
                    if isinstance(value, torch.Tensor):
                        json_data[key] = value.cpu().numpy().tolist()
                    else:
                        json_data[key] = value
                
                with open(filepath, 'w') as f:
                    json.dump(json_data, f, indent=2)
        
        elif format == 'pickle':
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(predictions, f)
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        self.logger.info(f"Saved predictions to {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        if self.model is None:
            return {'model_loaded': False}
        
        info = {
            'model_loaded': True,
            'model_type': self.model.__class__.__name__,
            'device': str(self.device),
            'model_summary': self.model.get_model_summary()
        }
        
        if self.preprocessor is not None:
            info['preprocessor_info'] = self.preprocessor.get_scaler_info()
        
        return info 