#!/usr/bin/env python3
"""Main training script for the PyTorch Training and Inference Pipeline."""

import argparse
import sys
import os
import torch
import logging
from pathlib import Path

# Set PyTorch to use double precision (float64) everywhere
torch.set_default_dtype(torch.float64)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import ConfigManager
from src.utils.logging import setup_logging
from src.data.dataloader import HDF5DataLoader
from src.data.preprocessing import DataPreprocessor
from src.models.mlp import MLP
from src.training.trainer import Trainer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train PyTorch model with HDF5 data')
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (auto, cpu, cuda, cuda:0, etc.)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Setup everything but do not start training'
    )
    
    return parser.parse_args()


def setup_device(device_str: str) -> torch.device:
    """Setup computation device."""
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    if device.type == 'cuda' and not torch.cuda.is_available():
        print(f"Warning: CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    
    return device


def setup_reproducibility(config: ConfigManager, logger: logging.Logger):
    """Setup reproducibility settings."""
    experiment_config = config.get('experiment', {})
    random_seed = experiment_config.get('random_seed', 42)
    deterministic = experiment_config.get('deterministic', True)
    
    # Set random seeds
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    
    # Set deterministic behavior
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        
        logger.info(f"Reproducibility enabled with seed: {random_seed}")


def create_model(config: ConfigManager, data_info: dict) -> MLP:
    """Create and initialize model."""
    model_config = config.get_model_config().copy()
    
    # Set input/output dimensions from data if not specified
    if model_config.get('input_dim') is None:
        model_config['input_dim'] = data_info['input_dim']
    
    if model_config.get('output_dim') is None:
        model_config['output_dim'] = data_info['output_dim']
    
    # Create model
    model_type = model_config.get('model_type', 'mlp')
    if model_type == 'mlp':
        model = MLP(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting PyTorch Training Pipeline")
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = ConfigManager(args.config)
        
        # Setup device
        device = setup_device(args.device)
        logger.info(f"Using device: {device}")
        
        # Setup reproducibility
        setup_reproducibility(config, logger)
        
        # Load data
        logger.info("Setting up data loading...")
        data_config = config.get_data_config()
        
        # Setup preprocessing
        preprocessing_config = data_config.get('preprocessing', {})
        preprocessor = None
        if preprocessing_config.get('method', 'standard') != 'none':
            logger.info("Setting up data preprocessing...")
            preprocessor = DataPreprocessor(method=preprocessing_config['method'])
        
        data_loader = HDF5DataLoader(
            input_file=data_config['input_file'],
            output_file=data_config['output_file'],
            preprocessor=preprocessor,
            v_start_kms=data_config.get('v_start_kms', 3000),
            v0_kms=data_config.get('v0_kms', 5000),
            t0_day=data_config.get('t0_day', 5),
            keep_v_outer=data_config.get('keep_v_outer', True),
            limit_nsamples=data_config.get('limit_nsamples', None)
        )
        
        # Create dataloaders
        train_loader, val_loader, test_loader = data_loader.load_and_create_dataloaders(
            train_split=data_config.get('train_split', 0.8),
            val_split=data_config.get('val_split', 0.1),
            test_split=data_config.get('test_split', 0.1),
            batch_size=data_config.get('batch_size', 32),
            shuffle=data_config.get('shuffle', True),
            num_workers=data_config.get('num_workers', 0),
            random_seed=data_config.get('random_seed', 42)
        )
        
        # Get data information
        data_info = data_loader.get_data_info()
        logger.info(f"Data loaded: {data_info['num_samples']} samples, "
                   f"{data_info['input_dim']} input features, "
                   f"{data_info['output_dim']} output features")
        
        # Save preprocessor if used
        if preprocessor is not None:
            preprocessor_path = os.path.join(
                config.get('training.checkpointing.checkpoint_dir', 'checkpoints'),
                'preprocessor.pkl'
            )
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            data_loader.save_preprocessor(preprocessor_path)
        
        # Create model
        logger.info("Creating model...")
        model = create_model(config, data_info)
        model.print_model_summary()
        
        # Prepare config for trainer with scaler for diagnostic plotting
        trainer_config = config.config.copy()
        
        # Add scalers to config for diagnostic plotting callbacks
        if preprocessor is not None and hasattr(preprocessor, 'scaler'):
            trainer_config['scaler'] = preprocessor.scaler
            # For pairwise analysis, we might need input scaler if inputs were preprocessed
            if hasattr(preprocessor, 'input_scaler'):
                trainer_config['input_scaler'] = preprocessor.input_scaler
            logger.info("Added scalers to config for diagnostic plotting")
        else:
            logger.info("No scalers available for diagnostic plotting")
        
        # Create trainer with callback system
        logger.info("Setting up trainer with callback system...")
        trainer = Trainer(model, trainer_config, device)
        
        # Log callback configuration
        training_config = config.get_training_config()
        logger.info("Callback configuration:")
        logger.info(f"  Early stopping: {training_config.get('early_stopping', {}).get('enabled', False)}")
        logger.info(f"  Checkpointing: {training_config.get('checkpointing', {}).get('enabled', True)}")
        logger.info(f"  Logging: {training_config.get('logging', {}).get('enabled', True)}")
        logger.info(f"  Training curves plotting: {training_config.get('diagnostic_plotting_curves', {}).get('enabled', True)}")
        logger.info(f"  Pairwise input analysis: {training_config.get('diagnostic_plotting_pairplot', {}).get('enabled', True)}")
        logger.info(f"  Gradient clipping: {training_config.get('gradient_clipping') is not None}")
        
        if args.dry_run:
            logger.info("Dry run complete. Exiting without training.")
            return
        
        # Start training
        epochs = training_config.get('epochs', 100)
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            resume_from_checkpoint=args.resume
        )
        
        # Evaluate on test set
        if test_loader is not None and len(test_loader.dataset) > 0:
            logger.info("Evaluating on test set...")
            test_results = trainer.test(test_loader)
            logger.info("Test evaluation completed")
        else:
            logger.info("No test data available, skipping test evaluation")
        
        # Print training summary
        summary = trainer.get_training_summary()
        logger.info("Training Summary:")
        logger.info(f"  Best validation score: {summary['best_val_score']:.6f}")
        logger.info(f"  Best epoch: {summary['best_epoch']}")
        logger.info(f"  Total epochs: {summary['total_epochs']}")
        
        # Log output directories
        logger.info("Training completed successfully!")
        logger.info("Check the following directories for outputs:")
        logger.info(f"  - Training curves plots: {training_config.get('diagnostic_plotting_curves', {}).get('plot_dir', 'plots')}")
        logger.info(f"  - Pairwise analysis plots: {training_config.get('diagnostic_plotting_pairplot', {}).get('plot_dir', 'plots')}")
        logger.info(f"  - Logs: {training_config.get('logging', {}).get('log_dir', 'logs')}")
        logger.info(f"  - Checkpoints: {training_config.get('checkpointing', {}).get('checkpoint_dir', 'checkpoints')}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == '__main__':
    main()
