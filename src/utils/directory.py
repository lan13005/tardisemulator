"""Directory management utilities for the TARDIS emulator training system."""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Union


class DirectoryManager:
    """Manages directories for the TARDIS emulator training system.
    
    This class creates and manages the main directories (plots, checkpoints, logs)
    relative to a root directory, making it easy to organize experiments and trials.
    
    Attributes:
        root_dir (Path): Root directory for all experiment outputs
        checkpoints_dir (Path): Directory for model checkpoints
        logs_dir (Path): Directory for log files and MLflow logs
        plots_dir (Path): Directory for diagnostic plots
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        create_dirs: bool = True
    ):
        """Initialize DirectoryManager.
        
        Args:
            root_dir: Root directory for all experiment outputs
            create_dirs: Whether to create directories immediately
        """
        self.root_dir = Path(root_dir)
        self.logger = logging.getLogger('pytorch_pipeline')
        
        # Define main subdirectories
        self.checkpoints_dir = self.root_dir / "checkpoints"
        self.logs_dir = self.root_dir / "logs"
        self.plots_dir = self.root_dir / "plots"
        
        # Create directories if requested
        if create_dirs:
            self.create_directories()
    
    def create_directories(self) -> None:
        """Create all necessary directories."""
        directories = [
            self.root_dir,
            self.checkpoints_dir,
            self.logs_dir,
            self.plots_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Created directories under: {self.root_dir}")
    
    def get_directory_paths(self) -> Dict[str, Path]:
        """Get all directory paths as a dictionary.
        
        Returns:
            Dictionary mapping directory names to their paths
        """
        return {
            'root': self.root_dir,
            'checkpoints': self.checkpoints_dir,
            'logs': self.logs_dir,
            'plots': self.plots_dir
        }
    
    def get_relative_paths(self) -> Dict[str, str]:
        """Get all directory paths as relative strings from root.
        
        Returns:
            Dictionary mapping directory names to their relative paths
        """
        paths = self.get_directory_paths()
        return {name: str(path.relative_to(self.root_dir)) for name, path in paths.items()}
    
    def update_config_paths(self, config: Dict) -> Dict:
        """Update configuration dictionary with relative paths.
        
        Args:
            config: Configuration dictionary to update
            
        Returns:
            Updated configuration dictionary
        """
        # Create a copy to avoid modifying the original
        updated_config = config.copy()
        
        # Update training configuration paths
        if 'training' in updated_config:
            training_config = updated_config['training']
            
            # Update checkpointing path
            if 'checkpointing' in training_config:
                training_config['checkpointing']['checkpoint_dir'] = str(self.checkpoints_dir)
            
            # Update logging path
            if 'logging' in training_config:
                training_config['logging']['log_dir'] = str(self.logs_dir)
            
            # Update diagnostic plotting paths
            if 'diagnostic_plotting_curves' in training_config:
                training_config['diagnostic_plotting_curves']['plot_dir'] = str(self.plots_dir)
            
            if 'diagnostic_plotting_pairplot' in training_config:
                training_config['diagnostic_plotting_pairplot']['plot_dir'] = str(self.plots_dir)
        
        return updated_config
    
    def create_trial_directory(self, trial_name: str) -> 'DirectoryManager':
        """Create a subdirectory for a specific trial (useful for Optuna).
        
        Args:
            trial_name: Name of the trial
            
        Returns:
            New DirectoryManager instance for the trial
        """
        trial_dir = self.root_dir / trial_name
        return DirectoryManager(trial_dir, create_dirs=True)
    
    def __str__(self) -> str:
        """String representation of the DirectoryManager."""
        return f"DirectoryManager(root={self.root_dir})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the DirectoryManager."""
        paths = self.get_directory_paths()
        paths_str = "\n".join([f"  {name}: {path}" for name, path in paths.items()])
        return f"DirectoryManager(\nroot_dir={self.root_dir}\ndirectories:\n{paths_str}\n)"
