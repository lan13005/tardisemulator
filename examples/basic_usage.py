#!/usr/bin/env python3
"""Basic usage example of the PyTorch Training and Inference Pipeline."""

import sys
import os
import torch

# Set PyTorch to use double precision (float64) everywhere
torch.set_default_dtype(torch.float64)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.config import ConfigManager
from src.utils.logging import setup_logging
from src.data.dataloader import HDF5DataLoader
from src.data.preprocessing import DataPreprocessor
from src.models.mlp import MLP
from src.training.trainer import Trainer
from src.inference.predictor import Predictor


def main():
    """Basic usage example."""
    # Setup logging
    logger = setup_logging('INFO')
    logger.info("Basic Usage Example")
    
    # 1. Configuration Management
    logger.info("=== Configuration Management ===")
    
    # Load configuration
    config = ConfigManager('src/configs/experiment_configs/small_test.yaml')
    
    # Access configuration values
    batch_size = config.get('data.batch_size', 32)
    learning_rate = config.get('training.optimizer.learning_rate', 0.001)
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    
    # 2. Data Loading
    logger.info("\n=== Data Loading ===")
    
    # Check if data files exist
    data_config = config.get_data_config()
    if not os.path.exists(data_config['input_file']):
        logger.warning(f"Input file not found: {data_config['input_file']}")
        logger.info("Please ensure the HDF5 data files are in the inputs/ directory")
        return
    
    # Create data loader
    data_loader = HDF5DataLoader(
        input_file=data_config['input_file'],
        output_file=data_config['output_file'],
        v_start_kms=data_config.get('v_start_kms', 3000),
        v0_kms=data_config.get('v0_kms', 5000),
        t0_day=data_config.get('t0_day', 5),
        keep_v_outer=data_config.get('keep_v_outer', True)
    )
    
    # Load data and create dataloaders
    train_loader, val_loader, test_loader = data_loader.load_and_create_dataloaders(
        train_split=0.7,
        val_split=0.2,
        test_split=0.1,
        batch_size=16,
        random_seed=42
    )
    
    # Get data information
    data_info = data_loader.get_data_info()
    logger.info(f"Data shape: input={data_info['input_shape']}, output={data_info['output_shape']}")
    logger.info(f"Features: {len(data_info['feature_names'])}")
    
    # 3. Data Preprocessing
    logger.info("\n=== Data Preprocessing ===")
    
    # Create and fit preprocessor
    preprocessor = DataPreprocessor(method='standard')
    
    # Get some training data to fit preprocessor
    for batch_inputs, batch_outputs in train_loader:
        preprocessor.fit(batch_inputs, batch_outputs)
        break  # Just use first batch for example
    
    # Transform data
    transformed_inputs = preprocessor.transform_input(batch_inputs)
    logger.info(f"Original data range: [{batch_inputs.min():.3f}, {batch_inputs.max():.3f}]")
    logger.info(f"Transformed data range: [{transformed_inputs.min():.3f}, {transformed_inputs.max():.3f}]")
    
    # 4. Model Creation
    logger.info("\n=== Model Creation ===")
    
    # Create model configuration
    model_config = {
        'model_type': 'mlp',
        'input_dim': data_info['input_dim'],
        'output_dim': data_info['output_dim'],
        'hidden_dims': [64, 32],
        'activation': 'relu',
        'dropout_rate': 0.1,
        'use_batch_norm': False,
        'use_bias': True
    }
    
    # Create model
    model = MLP(model_config)
    model.print_model_summary()
    
    # 5. Training Setup
    logger.info("\n=== Training Setup ===")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create trainer with minimal configuration
    training_config = {
        'training': {
            'task_type': 'regression',
            'epochs': 3,  # Very few epochs for example
            'loss': {'type': 'mse'},
            'optimizer': {
                'type': 'adam',
                'learning_rate': 0.01
            },
            'scheduler': {'type': 'none'},
            'early_stopping': {'enabled': False},
            'checkpointing': {
                'enabled': True,
                'checkpoint_dir': 'examples/checkpoints'
            },
            'logging': {
                'enabled': True,
                'log_dir': 'examples/logs',
                'experiment_name': 'basic_example',
                'use_tensorboard': False
            }
        }
    }
    
    trainer = Trainer(model, training_config, device)
    
    # 6. Training
    logger.info("\n=== Training ===")
    
    # Train for a few epochs
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3
    )
    
    logger.info("Training completed!")
    logger.info(f"Final train loss: {history['history']['train_loss'][-1]:.6f}")
    if history['history']['val_loss'][-1] is not None:
        logger.info(f"Final val loss: {history['history']['val_loss'][-1]:.6f}")
    
    # 7. Inference
    logger.info("\n=== Inference ===")
    
    # Create predictor
    predictor = Predictor(model=model, preprocessor=preprocessor, device=device)
    
    # Make predictions on test data
    for test_inputs, test_targets in test_loader:
        predictions = predictor.predict(test_inputs)
        
        logger.info(f"Test batch shape: {test_inputs.shape}")
        logger.info(f"Predictions shape: {predictions['predictions'].shape}")
        logger.info(f"Prediction range: [{predictions['predictions'].min():.3f}, {predictions['predictions'].max():.3f}]")
        logger.info(f"Target range: [{test_targets.min():.3f}, {test_targets.max():.3f}]")
        break  # Just show first batch
    
    # 8. Model Evaluation
    logger.info("\n=== Model Evaluation ===")
    
    # Evaluate on test set
    if test_loader:
        test_results = trainer.test(test_loader)
        logger.info("Test Results:")
        for metric, value in test_results.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.6f}")
    
    logger.info("\n=== Example Completed ===")
    logger.info("This example demonstrated:")
    logger.info("- Configuration management")
    logger.info("- Data loading and preprocessing")
    logger.info("- Model creation and training")
    logger.info("- Inference and evaluation")


if __name__ == '__main__':
    main() 