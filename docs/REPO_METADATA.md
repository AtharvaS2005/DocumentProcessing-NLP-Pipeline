# Repository Metadata

- Project: Image Denoising Pipeline with Swin-Unet
- Language: Python 3.11
- Entry points:
  - Web UI: `python main.py --ui`
  - Pipeline CLI: `python main.py --mode pipeline --train_dir input/train --clean_dir input/clean --test_dir input/test --epochs 80 --device cuda`
  - Train only: `python main.py --mode train --data_dir output/synthetic --epochs 80 --device cuda`
  - Test only: `python main.py --mode test --model_path output/models/best_model.pth --test_dir input/test --device cuda`
- Model config (default): img_size 256, window_size 8, depths [2, 2, 6, 2], heads [3, 6, 12, 24], embed_dim 96, losses = L1/Charbonnier/SSIM/Edge (0.30/0.30/0.25/0.15)
- Data layout: `input/{train,clean,test}`; outputs in `output/{synthetic,models,results}`
- Do not commit: virtual environments (`.venv*/`), large data (`input/`, `output/`), checkpoints (`*.pth`), logs (`logs/`), cache directories (`__pycache__/`, `.pytest_cache/`)
- Safe to commit: `src/`, `requirements.txt`, `README.md`, `QUICKSTART.md`, this metadata file, config/scripts, small sample data (if under a few MB)
- License: MIT (see README)
- Repro steps:
  1) `python -m venv .venv311 && .\.venv311\Scripts\Activate.ps1`
  2) `pip install -r requirements.txt`
  3) Launch UI or run pipeline commands above
- TensorBoard logs: `logs/` (ignored by default)
- Notes: When training on dotted text noise, prefer 80–120 epochs and include salt-and-pepper noise in synthetic generation.
