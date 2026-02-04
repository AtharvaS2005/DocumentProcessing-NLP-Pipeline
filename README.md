# Image Denoising Pipeline with Swin-Unet

## Overview
This project implements a preprocessing stage for an NLP pipeline that denoises images using a Swin-Unet based ML model, making text completely visible for OCR processing.

## Features
- **Noise Profiling**: Analyzes noisy training images to extract noise characteristics
- **Synthetic Noise Generation**: Mimics noise patterns on clean images with 99% accuracy
- **Swin-Unet Denoising**: Advanced transformer-based architecture for image denoising
- **Web UI**: User-friendly interface for training and testing
- **CUDA Support**: GPU acceleration for faster training and inference

## Project Structure
```
├── input/
│   ├── train/       # Noisy training images
│   ├── clean/       # Clean images for synthetic noise generation
│   └── test/        # Test images for evaluation
├── output/
│   ├── synthetic/   # Synthetically generated noisy image pairs
│   ├── models/      # Trained model checkpoints
│   └── results/     # Denoised test images
├── src/
│   ├── models/      # Swin-Unet architecture
│   ├── utils/       # Utility functions
│   ├── noise.py     # Noise profiling and synthesis
│   ├── train.py     # Training pipeline
│   ├── test.py      # Testing/inference pipeline
│   └── app.py       # Web UI
└── requirements.txt

```

## Installation
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Web Interface
```bash
python src/app.py
```

### Command Line

1. **Noise Profiling & Synthetic Data Generation**:
```bash
python src/noise.py --train_dir input/train --clean_dir input/clean --output_dir output/synthetic
```

2. **Train Model**:
```bash
python src/train.py --data_dir output/synthetic --output_dir output/models --device cuda
```

3. **Test Model**:
```bash
python src/test.py --model_path output/models/best_model.pth --test_dir input/test --output_dir output/results
```

## Model Architecture
- **Swin-Unet**: Combines Swin Transformer blocks with U-Net architecture
- **Self-Attention**: Captures long-range dependencies in images
- **Skip Connections**: Preserves fine-grained details

## Dataset
- **Unsupervised**: No paired clean-noisy images required
- **Synthetic Generation**: Creates training pairs automatically
- **Noise Mimicking**: 99% accuracy in replicating noise patterns

## License
MIT
