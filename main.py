"""
Main execution script for the image denoising pipeline.
Can run the complete pipeline from command line or launch the web UI.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.noise import NoiseProfiler, NoiseSynthesizer
from src.app import main as launch_ui


def run_complete_pipeline(
    train_dir: str,
    clean_dir: str,
    test_dir: str,
    output_dir: str,
    epochs: int,
    device: str,
    batch_size: int
):
    """
    Run the complete denoising pipeline from command line.
    
    Pipeline steps:
    1. Profile noise from training images
    2. Generate synthetic noisy dataset
    3. Train denoising model
    4. Test on test images
    """
    import torch
    from src.models.swin_unet import SwinUnet
    from src.utils.data_utils import create_dataloaders
    from src.train import DenoisingTrainer
    from src.test import ImageDenoiser
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    synthetic_dir = output_path / 'synthetic'
    models_dir = output_path / 'models'
    results_dir = output_path / 'results'
    
    print("=" * 70)
    print("IMAGE DENOISING PIPELINE - COMPLETE WORKFLOW")
    print("=" * 70)
    
    # Step 1: Noise Profiling
    print("\n[1/4] NOISE PROFILING")
    print("-" * 70)
    profiler = NoiseProfiler()
    profiler.profile_directory(
        train_dir,
        str(output_path / 'noise_profile.json')
    )
    print(f"✓ Profiled {len(profiler.noise_profiles)} training images")
    
    # Step 2: Synthetic Data Generation
    print("\n[2/4] SYNTHETIC DATA GENERATION")
    print("-" * 70)
    synthesizer = NoiseSynthesizer(profiler)
    count = synthesizer.generate_synthetic_dataset(
        clean_dir,
        str(synthetic_dir),
        use_random_profiles=True
    )
    print(f"✓ Generated {count} synthetic image pairs")
    
    # Step 3: Model Training
    print("\n[3/4] MODEL TRAINING")
    print("-" * 70)
    
    # Set device
    device_str = 'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device_str}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir=str(synthetic_dir),
        batch_size=batch_size,
        image_size=256,
        train_split=0.9,
        num_workers=0
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    model = SwinUnet(
        img_size=256,
        patch_size=4,
        in_chans=3,
        out_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=8
    )
    
    # Train
    trainer = DenoisingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device_str,
        learning_rate=1e-4,
        output_dir=str(models_dir),
        log_dir='logs'
    )
    
    trainer.train(num_epochs=epochs)
    
    # Step 4: Testing
    print("\n[4/4] TESTING")
    print("-" * 70)
    
    model_path = str(models_dir / 'best_model.pth')
    denoiser = ImageDenoiser(
        model_path=model_path,
        device=device_str,
        image_size=256
    )
    
    denoiser.denoise_directory(
        input_dir=test_dir,
        output_dir=str(results_dir),
        save_comparison=True
    )
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📊 Results:")
    print(f"  • Synthetic dataset: {synthetic_dir}")
    print(f"  • Trained model: {model_path}")
    print(f"  • Denoised images: {results_dir}")
    print(f"  • Best PSNR: {trainer.best_psnr:.2f} dB")
    print("\n✓ All outputs saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Image Denoising Pipeline with Swin-Unet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch web UI
  python main.py --ui
  
  # Run complete pipeline
  python main.py --mode pipeline --train_dir input/train --clean_dir input/clean --test_dir input/test
  
  # Just generate synthetic data
  python main.py --mode synthetic --train_dir input/train --clean_dir input/clean
  
  # Just train model
  python main.py --mode train --data_dir output/synthetic --epochs 50
  
  # Just test model
  python main.py --mode test --model_path output/models/best_model.pth --test_dir input/test
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['pipeline', 'synthetic', 'train', 'test'],
        help='Execution mode'
    )
    parser.add_argument('--ui', action='store_true', help='Launch web UI')
    parser.add_argument('--train_dir', type=str, default='input/train',
                        help='Directory with noisy training images')
    parser.add_argument('--clean_dir', type=str, default='input/clean',
                        help='Directory with clean images')
    parser.add_argument('--test_dir', type=str, default='input/test',
                        help='Directory with test images')
    parser.add_argument('--data_dir', type=str, default='output/synthetic',
                        help='Directory with synthetic dataset')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--model_path', type=str, default='output/models/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Launch UI if requested
    if args.ui:
        print("Launching web UI...")
        launch_ui()
        return
    
    # Execute based on mode
    if args.mode == 'pipeline':
        run_complete_pipeline(
            train_dir=args.train_dir,
            clean_dir=args.clean_dir,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            device=args.device,
            batch_size=args.batch_size
        )
    
    elif args.mode == 'synthetic':
        print("Generating synthetic data...")
        profiler = NoiseProfiler()
        profiler.profile_directory(args.train_dir, 'output/noise_profile.json')
        synthesizer = NoiseSynthesizer(profiler)
        synthesizer.generate_synthetic_dataset(
            args.clean_dir,
            args.data_dir,
            use_random_profiles=True
        )
    
    elif args.mode == 'train':
        print("Training model...")
        import torch
        from src.models.swin_unet import SwinUnet
        from src.utils.data_utils import create_dataloaders
        from src.train import DenoisingTrainer
        
        device_str = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
        
        train_loader, val_loader = create_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=256,
            train_split=0.9,
            num_workers=0
        )
        
        model = SwinUnet(
            img_size=256,
            patch_size=4,
            in_chans=3,
            out_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=8
        )
        
        trainer = DenoisingTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device_str,
            output_dir='output/models'
        )
        
        trainer.train(num_epochs=args.epochs)
    
    elif args.mode == 'test':
        print("Testing model...")
        from src.test import ImageDenoiser
        
        device_str = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
        
        denoiser = ImageDenoiser(
            model_path=args.model_path,
            device=device_str,
            image_size=256,
            window_size=8,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            embed_dim=96
        )
        
        denoiser.denoise_directory(
            input_dir=args.test_dir,
            output_dir='output/results',
            save_comparison=True
        )
    
    else:
        # Default: show help and suggest UI
        parser.print_help()
        print("\n💡 Tip: Run 'python main.py --ui' to launch the web interface!")


if __name__ == "__main__":
    main()
