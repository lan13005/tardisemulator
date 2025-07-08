#!/usr/bin/env python3
"""Inference script for making predictions with trained models."""

import argparse
import sys
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Set PyTorch to use double precision (float64) everywhere
torch.set_default_dtype(torch.float64)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import ConfigManager
from src.utils.logging import setup_logging
from src.inference.predictor import Predictor
from src.data.preprocessing import DataPreprocessor


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    
    parser.add_argument(
        '--model-path', '-m',
        type=str,
        required=True,
        help='Path to trained model file'
    )
    
    parser.add_argument(
        '--input-file', '-i',
        type=str,
        required=True,
        help='Path to input data file (HDF5, CSV, or JSON)'
    )
    
    parser.add_argument(
        '--output-file', '-o',
        type=str,
        required=True,
        help='Path to save predictions'
    )
    
    parser.add_argument(
        '--config-path', '-c',
        type=str,
        default=None,
        help='Path to model configuration file'
    )
    
    parser.add_argument(
        '--preprocessor-path', '-p',
        type=str,
        default=None,
        help='Path to saved preprocessor'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=64,
        help='Batch size for inference'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (auto, cpu, cuda, cuda:0, etc.)'
    )
    
    parser.add_argument(
        '--output-format',
        type=str,
        default='csv',
        choices=['csv', 'json', 'pickle'],
        help='Output format'
    )
    
    parser.add_argument(
        '--return-probabilities',
        action='store_true',
        help='Return class probabilities for classification tasks'
    )
    
    parser.add_argument(
        '--no-preprocessing',
        action='store_true',
        help='Skip preprocessing'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
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


def load_input_data(input_file: str) -> pd.DataFrame:
    """Load input data from file."""
    file_ext = Path(input_file).suffix.lower()
    
    if file_ext == '.h5' or file_ext == '.hdf5':
        # For HDF5 files, try to read as pandas
        df = pd.read_hdf(input_file)
    elif file_ext == '.csv':
        df = pd.read_csv(input_file)
    elif file_ext == '.json':
        df = pd.read_json(input_file)
    elif file_ext == '.pkl' or file_ext == '.pickle':
        df = pd.read_pickle(input_file)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    return df


def main():
    """Main inference function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting PyTorch Inference Pipeline")
    
    try:
        # Setup device
        device = setup_device(args.device)
        logger.info(f"Using device: {device}")
        
        # Load input data
        logger.info(f"Loading input data from: {args.input_file}")
        input_data = load_input_data(args.input_file)
        logger.info(f"Loaded data: {input_data.shape}")
        
        # Setup preprocessor
        preprocessor = None
        if not args.no_preprocessing and args.preprocessor_path:
            logger.info(f"Loading preprocessor from: {args.preprocessor_path}")
            preprocessor = DataPreprocessor()
            preprocessor.load_scalers(args.preprocessor_path)
        
        # Create predictor
        logger.info(f"Loading model from: {args.model_path}")
        predictor = Predictor(
            model_path=args.model_path,
            config_path=args.config_path,
            device=device,
            preprocessor=preprocessor
        )
        
        # Print model info
        model_info = predictor.get_model_info()
        logger.info("Model Information:")
        logger.info(f"  Model type: {model_info.get('model_type', 'Unknown')}")
        logger.info(f"  Device: {model_info.get('device', 'Unknown')}")
        if 'model_summary' in model_info:
            summary = model_info['model_summary']
            logger.info(f"  Parameters: {summary.get('total_parameters', 0):,}")
            logger.info(f"  Input dim: {summary.get('input_dim', 'Unknown')}")
            logger.info(f"  Output dim: {summary.get('output_dim', 'Unknown')}")
        
        # Make predictions
        logger.info("Making predictions...")
        predictions_df = predictor.predict_dataframe(
            df=input_data,
            apply_preprocessing=not args.no_preprocessing,
            return_probabilities=args.return_probabilities,
            batch_size=args.batch_size
        )
        
        logger.info(f"Generated predictions: {predictions_df.shape}")
        
        # Save predictions
        logger.info(f"Saving predictions to: {args.output_file}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        
        if args.output_format == 'csv':
            predictions_df.to_csv(args.output_file, index=False)
        elif args.output_format == 'json':
            predictions_df.to_json(args.output_file, orient='records', indent=2)
        elif args.output_format == 'pickle':
            predictions_df.to_pickle(args.output_file)
        
        logger.info("Inference completed successfully!")
        
        # Print summary statistics
        logger.info("Prediction Summary:")
        for col in predictions_df.columns:
            if predictions_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                logger.info(f"  {col}: mean={predictions_df[col].mean():.6f}, "
                           f"std={predictions_df[col].std():.6f}, "
                           f"min={predictions_df[col].min():.6f}, "
                           f"max={predictions_df[col].max():.6f}")
        
    except Exception as e:
        logger.error(f"Inference failed with error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == '__main__':
    main() 