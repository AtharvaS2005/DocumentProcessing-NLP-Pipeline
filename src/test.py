"""
Testing/inference pipeline for Swin-Unet denoising model.
"""

import os
import torch
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from torchvision import transforms

try:
    from .models.swin_unet import SwinUnet
    from .utils.data_utils import calculate_psnr, calculate_ssim
except ImportError:
    from models.swin_unet import SwinUnet
    from utils.data_utils import calculate_psnr, calculate_ssim


class ImageDenoiser:
    """Image denoising inference class"""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        image_size: int = 256,
        window_size: int = 8,
        embed_dim: int = 96,
        depths: list | None = None,
        num_heads: list | None = None,
    ):
        """
        Initialize the denoiser.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            image_size: Size to process images at
            window_size: Swin window size (8 pairs well with 256px)
            embed_dim: Base embedding dimension
            depths: Transformer depth per stage
            num_heads: Attention heads per stage
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu' and device == 'cuda':
            print("CUDA not available, using CPU for inference")
        
        self.image_size = image_size
        self.window_size = window_size
        self.depths = depths or [2, 2, 6, 2]
        self.num_heads = num_heads or [3, 6, 12, 24]
        self.embed_dim = embed_dim
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = SwinUnet(
            img_size=image_size,
            patch_size=4,
            in_chans=3,
            out_chans=3,
            embed_dim=self.embed_dim,
            depths=self.depths,
            num_heads=self.num_heads,
            window_size=self.window_size
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        
        # Transforms
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
    
    def denoise_image(self, image_path: str, return_metrics: bool = False):
        """
        Denoise a single image.
        
        Args:
            image_path: Path to noisy image
            return_metrics: Whether to return quality metrics
        
        Returns:
            denoised_image: Denoised PIL Image
            metrics (optional): Dict with PSNR and SSIM if reference available
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        
        # Resize to model input size
        img_resized = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Convert to tensor
        img_tensor = self.to_tensor(img_resized).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            denoised_tensor = self.model(img_tensor)
        
        # Convert back to PIL
        denoised_tensor = denoised_tensor.squeeze(0).cpu()
        denoised_tensor = torch.clamp(denoised_tensor, 0, 1)
        denoised_img = self.to_pil(denoised_tensor)
        
        # Resize back to original size
        denoised_img = denoised_img.resize(original_size, Image.LANCZOS)
        
        if return_metrics:
            # This would require a clean reference image
            return denoised_img, None
        
        return denoised_img
    
    def denoise_directory(
        self,
        input_dir: str,
        output_dir: str,
        save_comparison: bool = False
    ):
        """
        Denoise all images in a directory.
        
        Args:
            input_dir: Directory containing noisy images
            output_dir: Directory to save denoised images
            save_comparison: Whether to save side-by-side comparisons
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            raise ValueError(f"No images found in {input_dir}")
        
        print(f"Processing {len(image_files)} images...")
        
        for img_file in tqdm(image_files, desc="Denoising"):
            try:
                # Denoise image
                denoised_img = self.denoise_image(str(img_file))
                
                # Save denoised image
                output_file = output_path / img_file.name
                denoised_img.save(output_file)
                
                # Save comparison if requested
                if save_comparison:
                    self._save_comparison(
                        str(img_file),
                        denoised_img,
                        output_path / f'comparison_{img_file.name}'
                    )
            
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
        
        print(f"✓ Denoised images saved to {output_path}")
    
    def _save_comparison(self, noisy_path: str, denoised_img: Image, output_path: Path):
        """Save side-by-side comparison of noisy and denoised images"""
        noisy_img = Image.open(noisy_path).convert('RGB')
        
        # Ensure same size
        if noisy_img.size != denoised_img.size:
            denoised_img = denoised_img.resize(noisy_img.size, Image.LANCZOS)
        
        # Create side-by-side comparison
        width, height = noisy_img.size
        comparison = Image.new('RGB', (width * 2, height))
        comparison.paste(noisy_img, (0, 0))
        comparison.paste(denoised_img, (width, 0))
        
        comparison.save(output_path)
    
    def batch_denoise(self, image_list: list) -> list:
        """
        Denoise a batch of images.
        
        Args:
            image_list: List of image paths or PIL Images
        
        Returns:
            List of denoised PIL Images
        """
        denoised_images = []
        
        for img_input in tqdm(image_list, desc="Batch denoising"):
            if isinstance(img_input, str):
                denoised_img = self.denoise_image(img_input)
            elif isinstance(img_input, Image.Image):
                # Save temporarily and process
                temp_path = "temp_image.png"
                img_input.save(temp_path)
                denoised_img = self.denoise_image(temp_path)
                os.remove(temp_path)
            else:
                raise ValueError(f"Unsupported input type: {type(img_input)}")
            
            denoised_images.append(denoised_img)
        
        return denoised_images


def evaluate_denoising(
    model_path: str,
    test_dir: str,
    clean_dir: str = None,
    device: str = 'cuda'
):
    """
    Evaluate denoising performance with metrics.
    
    Args:
        model_path: Path to trained model
        test_dir: Directory with noisy test images
        clean_dir: Directory with corresponding clean images (if available)
        device: Device for inference
    """
    denoiser = ImageDenoiser(model_path, device)
    
    test_path = Path(test_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    test_files = [
        f for f in test_path.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if not test_files:
        raise ValueError(f"No images found in {test_dir}")
    
    print(f"Evaluating on {len(test_files)} images...")
    
    total_psnr = 0
    total_ssim = 0
    count = 0
    
    for test_file in tqdm(test_files, desc="Evaluating"):
        try:
            # Denoise
            denoised_img = denoiser.denoise_image(str(test_file))
            
            # Calculate metrics if clean reference is available
            if clean_dir:
                clean_path = Path(clean_dir) / test_file.name
                if clean_path.exists():
                    clean_img = Image.open(clean_path).convert('RGB')
                    
                    # Convert to tensors
                    denoised_tensor = denoiser.to_tensor(denoised_img).unsqueeze(0)
                    clean_tensor = denoiser.to_tensor(clean_img).unsqueeze(0)
                    
                    # Calculate metrics
                    psnr = calculate_psnr(denoised_tensor, clean_tensor)
                    ssim = calculate_ssim(denoised_tensor, clean_tensor)
                    
                    total_psnr += psnr
                    total_ssim += ssim
                    count += 1
        
        except Exception as e:
            print(f"Error evaluating {test_file}: {e}")
    
    if count > 0:
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        print(f"\nEvaluation Results:")
        print(f"  Average PSNR: {avg_psnr:.2f} dB")
        print(f"  Average SSIM: {avg_ssim:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Test Swin-Unet denoising model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--test_dir', type=str, default='input/test',
                        help='Directory with test images')
    parser.add_argument('--output_dir', type=str, default='output/results',
                        help='Directory to save denoised images')
    parser.add_argument('--clean_dir', type=str, default=None,
                        help='Directory with clean reference images for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for processing')
    parser.add_argument('--save_comparison', action='store_true',
                        help='Save side-by-side comparisons')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation with metrics')
    
    args = parser.parse_args()
    
    # Create denoiser
    denoiser = ImageDenoiser(
        model_path=args.model_path,
        device=args.device,
        image_size=args.image_size
    )
    
    # Denoise images
    denoiser.denoise_directory(
        input_dir=args.test_dir,
        output_dir=args.output_dir,
        save_comparison=args.save_comparison
    )
    
    # Evaluate if requested
    if args.evaluate and args.clean_dir:
        print("\nRunning evaluation...")
        evaluate_denoising(
            model_path=args.model_path,
            test_dir=args.test_dir,
            clean_dir=args.clean_dir,
            device=args.device
        )


if __name__ == "__main__":
    main()
