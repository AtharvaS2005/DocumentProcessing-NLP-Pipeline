# Quick Start Guide

## Installation

1. Create and activate virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

## Usage Options

### Option 1: Web UI (Recommended for Beginners)

Launch the interactive web interface:
```powershell
python main.py --ui
```

Then open your browser to `http://localhost:7860`

**Workflow in UI:**
1. **Data Preparation Tab:**
   - Upload noisy training images
   - Upload clean images
   - Click "Generate Synthetic Noisy Images"

2. **Training Tab:**
   - Select device (CPU/GPU)
   - Set epochs (recommended: 50)
   - Click "Start Training"

3. **Testing Tab:**
   - Upload test images
   - Click "Denoise Images"
   - View results in gallery

### Option 2: Command Line

**Complete Pipeline:**
```powershell
python main.py --mode pipeline --train_dir input/train --clean_dir input/clean --test_dir input/test --epochs 50 --device cuda
```

**Individual Steps:**

1. Generate synthetic data only:
```powershell
python main.py --mode synthetic --train_dir input/train --clean_dir input/clean
```

2. Train model only:
```powershell
python main.py --mode train --data_dir output/synthetic --epochs 50 --device cuda
```

3. Test model only:
```powershell
python main.py --mode test --model_path output/models/best_model.pth --test_dir input/test --device cuda
```

### Option 3: Using Individual Scripts

**Step 1 - Generate Synthetic Dataset:**
```powershell
python src/noise.py --train_dir input/train --clean_dir input/clean --output_dir output/synthetic
```

**Step 2 - Train Model:**
```powershell
python src/train.py --data_dir output/synthetic --output_dir output/models --epochs 50 --device cuda --batch_size 8
```

**Step 3 - Test Model:**
```powershell
python src/test.py --model_path output/models/best_model.pth --test_dir input/test --output_dir output/results --device cuda --save_comparison
```

## Directory Structure

```
input/
  ├── train/      # Place noisy training images here
  ├── clean/      # Place clean images here
  └── test/       # Place test images here

output/
  ├── synthetic/  # Generated synthetic dataset
  ├── models/     # Trained models
  └── results/    # Denoised test images
```

## Tips

- **GPU Acceleration**: Use `--device cuda` for faster training (requires CUDA-capable GPU)
- **Batch Size**: Reduce if you get out-of-memory errors (try 4 or 2)
- **Epochs**: Start with 50 epochs; increase for better quality
- **Image Formats**: Supports JPG, PNG, BMP, TIFF
- **Noise Accuracy**: Synthetic noise mimics training noise at ~99% accuracy

## Monitoring Training

View training progress with TensorBoard:
```powershell
tensorboard --logdir logs
```

Then open `http://localhost:6006`

## Troubleshooting

**CUDA not available:**
- Use `--device cpu` instead
- Or install PyTorch with CUDA support

**Out of memory:**
- Reduce batch size: `--batch_size 4`
- Reduce image size (edit in code)

**No images found:**
- Check that images are in correct directories
- Verify image file extensions (.jpg, .png, etc.)
