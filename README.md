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

## Quick Start

### 1. Prepare Your Data

Place your HDF5 files in the `inputs/` directory:
- `training_input_parameter_round18.h5` - Input parameters
- `training_output_spectra_round18.h5` - Output spectra

### 2. Train a Model

```bash
python train.py --config src/configs/default_config.yaml
```

### 3. Make Predictions

```bash
python predict.py \
    --model-path checkpoints/small_test/best_model.pt \
    --input-file inputs/training_input_parameter_round18.h5 \
    --output-file predictions/test_predictions.csv \
    --preprocessor-path checkpoints/small_test/preprocessor.pkl
``` 

## Usage Examples

### Training

#### Basic Training
```bash
python train.py --config src/configs/default_config.yaml
# python train.py --config src/configs/default_config.yaml --log-level DEBUG # debug mode
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
