"""HDF5 data loading utilities for PyTorch."""

import pandas as pd
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, Any
import logging
from .preprocessing import DataPreprocessor


def drop_tardis_grid_units(df):
    """Remove units from DataFrame columns that have astropy units."""
    df_no_units = df.copy()
    for col in df.columns:
        if hasattr(df[col].values[0], 'unit'):
            df_no_units[col] = df[col].values.value  # Remove units
    return df_no_units


def get_M_total_from_grid(df_parameters, v_start_kms, v0_kms, t0_day, v_outer_infinity=False):
    """Calculate total mass from grid parameters."""
    # This is a simplified version - the actual implementation would depend on the physics
    # For now, return a reasonable array of masses
    n_samples = len(df_parameters)
    return np.random.uniform(0.5, 2.0, n_samples)  # Placeholder implementation


def convert_unitless_tardis_grid_to_parameters(
    df_tardis_grid, v_start_kms, v0_kms, t0_day, keep_v_outer=False, logger: logging.Logger = None,
):
    """Convert TARDIS grid to parameter format.
    
    This function processes the raw TARDIS simulation data and converts it
    to a format suitable for machine learning training.
    
    Args:
        df_tardis_grid: DataFrame containing TARDIS simulation parameters
        v_start_kms: Starting velocity in km/s
        v0_kms: Reference velocity in km/s  
        t0_day: Reference time in days
        keep_v_outer: Whether to keep the outer velocity parameter
        
    Returns:
        Processed DataFrame with parameters suitable for ML training
    """
    
    if logger is None:
        logger = logging.getLogger('pytorch_pipeline')
    
    df_grid = df_tardis_grid.copy()
    
    # Drop units if present
    if hasattr(df_grid["model.structure.density.rho_0"].values[0], "unit"):
        df_grid = drop_tardis_grid_units(df_grid)
        logger.info("Dropping the units from the tardis grid")
    
    # Drop calculated columns that are not free parameters
    df_parameters = df_grid.drop(
        ["model.structure.velocity.num", "plasma.initial_t_inner"], axis=1
    )
    
    # Get mass fraction columns
    mass_fractions_columns = [
        col for col in df_parameters.columns if col.startswith("model.abundances.")
    ]
    logger.info(f"Converting to elemental mass and remove the rho_0 and v_outer parameter...")
    
    # Calculate total mass and get rid of rho_0 and v_outer
    if keep_v_outer:
        M_total_Msun = get_M_total_from_grid(df_parameters, v_start_kms, v0_kms, t0_day, v_outer_infinity=True)
    else:
        M_total_Msun = get_M_total_from_grid(df_parameters, v_start_kms, v0_kms, t0_day, v_outer_infinity=False)
    
    # Remove v_outer_kms if present
    if "v_outer_kms" in df_parameters.columns:
        df_parameters = df_parameters.drop(["v_outer_kms"], axis=1)
    
    # Convert mass fractions to absolute masses
    df_parameters[mass_fractions_columns] *= M_total_Msun.reshape(-1, 1)
    
    # Drop calculated columns and reorder based on keep_v_outer setting
    if keep_v_outer:
        df_parameters = df_parameters.drop(["model.structure.density.rho_0"], axis=1)
        
        if "zero_Ca_density" in df_parameters.columns:
            cols_params = df_parameters.columns.tolist()
            cols_params[0] = "supernova.time_explosion"
            cols_params[1] = "model.structure.velocity.stop"
            cols_params[2] = "model.structure.density.exponent"
            cols_params[3] = "supernova.luminosity_requested"
            cols_params[4] = "zero_Ca_density"
            df_parameters = df_parameters[cols_params]
            assert "zero_Ca_velocity_kms" not in df_parameters.columns, (
                "zero_Ca_velocity_kms and zero_Ca_density should not be set at the same time!"
            )
        
        if "zero_Ca_velocity_kms" in df_parameters.columns:
            cols_params = df_parameters.columns.tolist()
            cols_params[0] = "supernova.time_explosion"
            cols_params[1] = "model.structure.velocity.stop"
            cols_params[2] = "zero_Ca_velocity_kms"
            cols_params[3] = "model.structure.density.exponent"
            cols_params[4] = "supernova.luminosity_requested"
            df_parameters = df_parameters[cols_params]
    else:
        df_parameters = df_parameters.drop(
            ["model.structure.density.rho_0", "model.structure.velocity.stop"], axis=1
        )
        
        if "zero_Ca_density" in df_parameters.columns:
            cols_params = df_parameters.columns.tolist()
            cols_params[0] = "supernova.time_explosion"
            cols_params[1] = "model.structure.density.exponent"
            cols_params[2] = "supernova.luminosity_requested"
            cols_params[3] = "zero_Ca_density"
            df_parameters = df_parameters[cols_params]
            assert "zero_Ca_velocity_kms" not in df_parameters.columns, (
                "zero_Ca_velocity_kms and zero_Ca_density should not be set at the same time!"
            )
        
        if "zero_Ca_velocity_kms" in df_parameters.columns:
            cols_params = df_parameters.columns.tolist()
            cols_params[0] = "supernova.time_explosion"
            cols_params[1] = "zero_Ca_velocity_kms"
            cols_params[2] = "model.structure.density.exponent"
            cols_params[3] = "supernova.luminosity_requested"
            df_parameters = df_parameters[cols_params]

    # Rename mass fraction columns
    df_parameters = df_parameters.rename(
        columns={item: "mass_" + item.split(".")[-1] for item in mass_fractions_columns}
    )
    
    return df_parameters


class TardisDataset(Dataset):
    """PyTorch Dataset for TARDIS simulation data."""
    
    def __init__(
        self,
        input_data: torch.Tensor,
        output_data: torch.Tensor,
        transform: Optional[Any] = None,
        feature_names: Optional[list] = None,
        target_names: Optional[list] = None
    ):
        """Initialize TardisDataset.
        
        Args:
            input_data: Input features tensor
            output_data: Target values tensor
            transform: Optional data transformation
            feature_names: List of feature names
            target_names: List of target names
        """
        assert len(input_data) == len(output_data), "Input and output must have same length"
        
        self.input_data = input_data
        self.output_data = output_data
        self.transform = transform
        self.feature_names = feature_names
        self.target_names = target_names
        self.logger = logging.getLogger('pytorch_pipeline')
        
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.input_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item at index."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input_sample = self.input_data[idx]
        output_sample = self.output_data[idx]
        
        if self.transform:
            input_sample = self.transform(input_sample)
        
        return input_sample, output_sample


class HDF5DataLoader:
    """Data loader for HDF5 files with TARDIS-specific processing."""
    
    def __init__(
        self,
        input_file: str,
        output_file: str,
        preprocessor: Optional[DataPreprocessor] = None,
        v_start_kms: float = 3000,
        v0_kms: float = 5000,
        t0_day: float = 5,
        keep_v_outer: bool = True,
        limit_nsamples: int = None,
        log_scaling: bool = False,
    ):
        """Initialize HDF5DataLoader.
        
        Args:
            input_file: Path to input HDF5 file
            output_file: Path to output HDF5 file
            preprocessor: Optional DataPreprocessor instance for scaling data
            v_start_kms: Starting velocity parameter
            v0_kms: Reference velocity parameter
            t0_day: Reference time parameter
            keep_v_outer: Whether to keep outer velocity parameter
            limit_nsamples: Limit the number of samples to load
            log_scaling: Whether to apply log10 scaling to columns with index >= 3
        """
        self.input_file = input_file
        self.output_file = output_file
        self.preprocessor = preprocessor
        self.v_start_kms = v_start_kms
        self.v0_kms = v0_kms
        self.t0_day = t0_day
        self.keep_v_outer = keep_v_outer
        self.limit_nsamples = limit_nsamples
        self.log_scaling = log_scaling
        self.logger = logging.getLogger('pytorch_pipeline')
        
        # If no preprocessor provided, create one with log_scaling
        if self.preprocessor is None and log_scaling:
            self.preprocessor = DataPreprocessor(method='standard', log_scaling=log_scaling)
        
        # Data storage
        self.input_data = None
        self.output_data = None
        self.feature_names = None
        self.target_names = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and process data from HDF5 files.
        
        Returns:
            Tuple of (input_df, output_df)
        """
        self.logger.info(f"Loading input data from {self.input_file}")
        if self.limit_nsamples is not None:
            df_input = pd.read_hdf(self.input_file, stop=self.limit_nsamples)
        else:
            df_input = pd.read_hdf(self.input_file)
        
        self.logger.info(f"Loading output data from {self.output_file}")
        if self.limit_nsamples is not None:
            df_output = pd.read_hdf(self.output_file, stop=self.limit_nsamples)
        else:
            df_output = pd.read_hdf(self.output_file)
        
        # Process input data using TARDIS conversion
        self.logger.info("Processing TARDIS grid data")
        df_processed_input = convert_unitless_tardis_grid_to_parameters(
            df_input, 
            self.v_start_kms, 
            self.v0_kms, 
            self.t0_day, 
            keep_v_outer=self.keep_v_outer,
            logger=self.logger
        )
        
        # Store feature and target names
        self.feature_names = list(df_processed_input.columns)
        self.target_names = list(df_output.columns)
        
        self.logger.info(f"Input shape: {df_processed_input.shape}")
        self.logger.info(f"Output shape: {df_output.shape}")
        self.logger.info(f"Features: {self.feature_names}")
        
        return df_processed_input, df_output
    
    def create_tensors(
        self,
        input_df: pd.DataFrame,
        output_df: pd.DataFrame,
        dtype: torch.dtype = torch.float64
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert DataFrames to PyTorch tensors.
        
        Args:
            input_df: Input DataFrame
            output_df: Output DataFrame
            dtype: PyTorch data type
            
        Returns:
            Tuple of (input_tensor, output_tensor)
        """
        # Convert to numpy arrays first
        input_array = input_df.values.astype(np.float64)
        output_array = output_df.values.astype(np.float64)
        
        # Convert to tensors
        input_tensor = torch.tensor(input_array, dtype=dtype)
        output_tensor = torch.tensor(output_array, dtype=dtype)
        
        self.input_data = input_tensor
        self.output_data = output_tensor
        
        return input_tensor, output_tensor
    
    def create_dataset(
        self,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        transform: Optional[Any] = None
    ) -> TardisDataset:
        """Create PyTorch Dataset.
        
        Args:
            input_tensor: Input features tensor
            output_tensor: Target values tensor
            transform: Optional data transformation
            
        Returns:
            TardisDataset instance
        """
        return TardisDataset(input_tensor, output_tensor, transform, self.feature_names, self.target_names)
    
    def create_dataloader(
        self,
        dataset: TardisDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = False
    ) -> DataLoader:
        """Create PyTorch DataLoader.
        
        Args:
            dataset: TardisDataset instance
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            drop_last: Whether to drop last incomplete batch
            
        Returns:
            DataLoader instance
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last
        )
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about loaded data.
        
        Returns:
            Dictionary with data information
        """
        if self.input_data is None or self.output_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        info = {
            'input_shape': self.input_data.shape,
            'output_shape': self.output_data.shape,
            'input_dim': self.input_data.shape[1],
            'output_dim': self.output_data.shape[1],
            'num_samples': self.input_data.shape[0],
            'feature_names': self.feature_names,
            'target_names': self.target_names
        }
        
        # Add preprocessor information if available
        if self.preprocessor is not None and self.preprocessor.is_fitted:
            info['preprocessor'] = self.preprocessor.get_scaler_info()
        
        return info
    
    def save_preprocessor(self, filepath: str) -> None:
        """Save the fitted preprocessor to disk.
        
        Args:
            filepath: Path where to save the preprocessor
        """
        if self.preprocessor is not None and self.preprocessor.is_fitted:
            self.preprocessor.save_scalers(filepath)
            self.logger.info(f"Preprocessor saved to: {filepath}")
        else:
            self.logger.warning("No fitted preprocessor to save")
    
    def get_preprocessor(self) -> Optional[DataPreprocessor]:
        """Get the preprocessor instance.
        
        Returns:
            DataPreprocessor instance if available, None otherwise
        """
        return self.preprocessor
    
    def load_and_create_dataloaders(
        self,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        random_seed: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load data and create train/validation/test DataLoaders.
        
        Args:
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
            test_split: Proportion of data for testing
            batch_size: Batch size for all dataloaders
            shuffle: Whether to shuffle training data
            num_workers: Number of worker processes
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Validate splits
        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test splits must sum to 1.0")
        
        # Load and process data
        input_df, output_df = self.load_data()
        input_tensor, output_tensor = self.create_tensors(input_df, output_df)
        
        # Set random seed for reproducibility
        if random_seed is not None:
            torch.manual_seed(random_seed)
        
        # Calculate split indices
        n_samples = len(input_tensor)
        n_train = int(train_split * n_samples)
        n_val = int(val_split * n_samples)
        n_test = n_samples - n_train - n_val
        
        # Create random indices
        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # Split data
        train_input = input_tensor[train_indices]
        train_output = output_tensor[train_indices]
        
        # Handle empty splits gracefully
        if n_val > 0:
            val_input = input_tensor[val_indices]
            val_output = output_tensor[val_indices]
        else:
            # Create empty tensors with correct shape
            val_input = input_tensor[:0]  # Empty tensor with same number of features
            val_output = output_tensor[:0]  # Empty tensor with same number of features
            
        if n_test > 0:
            test_input = input_tensor[test_indices]
            test_output = output_tensor[test_indices]
        else:
            # Create empty tensors with correct shape
            test_input = input_tensor[:0]  # Empty tensor with same number of features
            test_output = output_tensor[:0]  # Empty tensor with same number of features
        
        # Apply preprocessing if preprocessor is provided
        if self.preprocessor is not None:
            self.logger.info("Applying data preprocessing...")
            
            # Fit preprocessor on training data only
            self.preprocessor.fit(train_input, train_output)
            self.logger.info(f"Preprocessor fitted with method: {self.preprocessor.method}")
            
            # Transform all splits
            train_input = self.preprocessor.transform_input(train_input)
            train_output = self.preprocessor.transform_output(train_output)
            val_input = self.preprocessor.transform_input(val_input)
            val_output = self.preprocessor.transform_output(val_output)
            test_input = self.preprocessor.transform_input(test_input)
            test_output = self.preprocessor.transform_output(test_output)
            
            self.logger.info("Data preprocessing completed")
        
        # Create datasets
        train_dataset = self.create_dataset(train_input, train_output)
        val_dataset = self.create_dataset(val_input, val_output)
        test_dataset = self.create_dataset(test_input, test_output)
        
        # Create dataloaders
        train_loader = self.create_dataloader(
            train_dataset, batch_size, shuffle=shuffle, num_workers=num_workers
        )
        
        # Handle empty validation dataset
        if len(val_dataset) > 0:
            val_loader = self.create_dataloader(
                val_dataset, batch_size, shuffle=False, num_workers=num_workers
            )
        else:
            val_loader = None
            self.logger.info("No validation data created (val_split=0)")
            
        # Handle empty test dataset
        if len(test_dataset) > 0:
            test_loader = self.create_dataloader(
                test_dataset, batch_size, shuffle=False, num_workers=num_workers
            )
        else:
            test_loader = None
            self.logger.info("No test data created (test_split=0)")
        
        self.logger.info(f"Created dataloaders: Train={len(train_dataset)}, "
                        f"Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        return train_loader, val_loader, test_loader 