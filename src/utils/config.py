"""Configuration management utilities."""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """Manages configuration loading, validation, and access."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize ConfigManager.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config_path = config_path
        self.config = {}
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Validate configuration
        self._validate_config()
        
        return self.config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key: Configuration key (supports dot notation like 'model.layers')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self, output_path: str) -> None:
        """Save current configuration to YAML file.
        
        Args:
            output_path: Path to save configuration
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_sections = ['data', 'model', 'training']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate data configuration
        data_config = self.config.get('data', {})
        if 'input_file' not in data_config:
            raise ValueError("Missing 'data.input_file' in configuration")
        if 'output_file' not in data_config:
            raise ValueError("Missing 'data.output_file' in configuration")
        
        # Validate model configuration
        model_config = self.config.get('model', {})
        if 'input_dim' not in model_config:
            raise ValueError("Missing 'model.input_dim' in configuration")
        if 'output_dim' not in model_config:
            raise ValueError("Missing 'model.output_dim' in configuration")
        
        # Validate training configuration
        training_config = self.config.get('training', {})
        if 'epochs' not in training_config:
            raise ValueError("Missing 'training.epochs' in configuration")
        if 'learning_rate' not in training_config:
            raise ValueError("Missing 'training.learning_rate' in configuration")
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration section."""
        return self.config.get('data', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration section."""
        return self.config.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration section."""
        return self.config.get('training', {})
    
    def get_inference_config(self) -> Dict[str, Any]:
        """Get inference configuration section."""
        return self.config.get('inference', {}) 