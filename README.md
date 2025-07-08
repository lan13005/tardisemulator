# PyTorch Training and Inference Pipeline

A comprehensive machine learning framework for training and inference with HDF5 data loading, configurable model architectures, and robust training/evaluation workflows. Specifically designed for TARDIS supernova simulation data but adaptable to other regression and classification tasks.

## Features

- **HDF5 Data Loading**: Efficient loading from HDF5 files with TARDIS-specific processing
- **Configurable Models**: MLP with various activation functions, dropout, and batch normalization
- **Flexible Training**: Multiple optimizers, schedulers, early stopping, and checkpointing
- **Comprehensive Evaluation**: Regression and classification metrics with TensorBoard integration
- **Easy Inference**: Standalone prediction pipeline with preprocessing
- **YAML Configuration**: All settings managed through configuration files

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TardisEmulator
```

2. Install dependencies:
```bash
uv add torch torchvision torchaudio pandas numpy scikit-learn h5py pyyaml tensorboard
```

## Quick Start

### 1. Prepare Your Data

Place your HDF5 files in the `inputs/` directory:
- `training_input_parameter_round18.h5` - Input parameters
- `training_output_spectra_round18.h5` - Output spectra

### 2. Train a Model

```bash
python train.py --config src/configs/experiment_configs/small_test.yaml
```

### 3. Make Predictions

```bash
python predict.py \
    --model-path checkpoints/small_test/best_model.pt \
    --input-file inputs/training_input_parameter_round18.h5 \
    --output-file predictions/test_predictions.csv \
    --preprocessor-path checkpoints/small_test/preprocessor.pkl
```

## Configuration

The framework uses YAML configuration files to manage all settings:

### Available Configurations

- `src/configs/default_config.yaml` - Default settings with all options
- `src/configs/experiment_configs/tardis_regression.yaml` - TARDIS spectral regression
- `src/configs/experiment_configs/small_test.yaml` - Quick testing configuration

### Configuration Structure

```yaml
# Data settings
data:
  input_file: "inputs/training_input_parameter_round18.h5"
  output_file: "inputs/training_output_spectra_round18.h5"
  batch_size: 32
  preprocessing:
    method: "standard"  # standard, minmax, robust, none

# Model settings
model:
  model_type: "mlp"
  hidden_dims: [512, 256, 128]
  activation: "relu"
  dropout_rate: 0.1
  use_batch_norm: true

# Training settings
training:
  epochs: 100
  task_type: "regression"
  optimizer:
    type: "adam"
    learning_rate: 0.001
  scheduler:
    type: "reduce_on_plateau"
  early_stopping:
    enabled: true
    patience: 20
```

## Usage Examples

### Training

#### Basic Training
```bash
python train.py --config src/configs/default_config.yaml
```

#### Resume Training
```bash
python train.py \
    --config src/configs/experiment_configs/tardis_regression.yaml \
    --resume checkpoints/tardis_regression/checkpoint_epoch_0050.pt
```

#### Custom Device
```bash
python train.py \
    --config src/configs/experiment_configs/small_test.yaml \
    --device cuda:1
```

### Inference

#### Basic Prediction
```bash
python predict.py \
    --model-path checkpoints/best_model.pt \
    --input-file data/test_input.h5 \
    --output-file predictions/results.csv
```

#### Batch Prediction
```bash
python predict.py \
    --model-path checkpoints/best_model.pt \
    --input-file data/large_dataset.h5 \
    --output-file predictions/results.json \
    --batch-size 128 \
    --output-format json
```

### Programmatic Usage

```python
from src.utils.config import ConfigManager
from src.data.dataloader import HDF5DataLoader
from src.models.mlp import MLP
from src.training.trainer import Trainer

# Load configuration
config = ConfigManager('src/configs/default_config.yaml')

# Setup data
data_loader = HDF5DataLoader(
    input_file='inputs/training_input_parameter_round18.h5',
    output_file='inputs/training_output_spectra_round18.h5'
)
train_loader, val_loader, test_loader = data_loader.load_and_create_dataloaders()

# Create model
model_config = config.get_model_config()
model_config['input_dim'] = data_loader.get_data_info()['input_dim']
model_config['output_dim'] = data_loader.get_data_info()['output_dim']
model = MLP(model_config)

# Train
trainer = Trainer(model, config.config)
history = trainer.train(train_loader, val_loader, epochs=100)
```

## Model Architecture

### MLP (Multi-Layer Perceptron)

Configurable neural network with:
- Variable number of hidden layers
- Multiple activation functions (ReLU, Tanh, Sigmoid, LeakyReLU, GELU, Swish)
- Dropout for regularization
- Batch normalization
- Flexible input/output dimensions

Example configuration:
```yaml
model:
  model_type: "mlp"
  input_dim: 20
  output_dim: 1000
  hidden_dims: [1024, 512, 256, 128, 64]
  activation: "relu"
  dropout_rate: 0.2
  use_batch_norm: true
  use_bias: true
```

## Training Features

### Optimizers
- Adam, AdamW, SGD, RMSprop, Adagrad, Adadelta
- Configurable parameters (learning rate, weight decay, momentum, etc.)

### Learning Rate Schedulers
- StepLR, MultiStepLR, ExponentialLR
- CosineAnnealingLR, CosineAnnealingWarmRestarts
- ReduceLROnPlateau, CyclicLR, OneCycleLR

### Training Features
- Early stopping with patience and metric monitoring
- Gradient clipping for training stability
- Automatic checkpointing with best model saving
- TensorBoard logging and visualization
- Comprehensive metrics (MSE, RMSE, MAE, R², etc.)

## Data Processing

### TARDIS-Specific Processing
The framework includes specialized processing for TARDIS simulation data:
- Unit handling and conversion
- Mass fraction to absolute mass conversion
- Parameter filtering and reordering
- Velocity and density calculations

### Preprocessing Options
- StandardScaler (zero mean, unit variance)
- MinMaxScaler (0-1 range)
- RobustScaler (median and IQR)
- Custom preprocessing pipelines

## Output and Results

### Training Outputs
- Model checkpoints in `checkpoints/`
- Training logs in `logs/`
- TensorBoard visualizations
- Training history and metrics

### Prediction Outputs
- CSV, JSON, or Pickle formats
- Configurable output structure
- Batch processing support
- Probability outputs for classification

## Directory Structure

```
TardisEmulator/
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Model architectures
│   ├── training/       # Training pipeline
│   ├── inference/      # Inference pipeline
│   ├── utils/          # Utilities (config, logging, checkpoints)
│   └── configs/        # Configuration files
├── inputs/             # Input data files
├── examples/           # Usage examples
├── train.py           # Main training script
├── predict.py         # Inference script
└── README.md
```

## Examples

See `examples/basic_usage.py` for a comprehensive example demonstrating:
- Configuration management
- Data loading and preprocessing
- Model creation and training
- Inference and evaluation

Run the example:
```bash
python examples/basic_usage.py
```

## Advanced Features

### Feature Importance
```python
from src.inference.predictor import Predictor

predictor = Predictor(model_path='best_model.pt')
importance = predictor.get_feature_importance(input_sample, method='gradient')
```

### Custom Loss Functions
```yaml
training:
  loss:
    type: "huber"
    delta: 1.0
```

### Mixed Precision Training
```yaml
experiment:
  use_amp: true
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in configuration
2. **Data loading errors**: Check HDF5 file paths and structure
3. **Training instability**: Enable gradient clipping or reduce learning rate
4. **Poor convergence**: Try different optimizers or learning rate schedules

### Debugging

Enable debug logging:
```bash
python train.py --config config.yaml --log-level DEBUG
```

Use the small test configuration for quick debugging:
```bash
python train.py --config src/configs/experiment_configs/small_test.yaml
```

