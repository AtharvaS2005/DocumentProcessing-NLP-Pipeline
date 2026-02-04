"""
Utility functions for data loading and image processing.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class DenoisingDataset(Dataset):
    """Dataset for paired clean-noisy images"""
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        augment: bool = True
    ):
        """
        Args:
            data_dir: Directory containing 'clean' and 'noisy' subdirectories
            image_size: Size to resize images to
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment = augment
        
        # Get image paths
        self.clean_dir = self.data_dir / 'clean'
        self.noisy_dir = self.data_dir / 'noisy'
        
        if not self.clean_dir.exists() or not self.noisy_dir.exists():
            raise ValueError(f"Clean and noisy directories must exist in {data_dir}")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.clean_images = sorted([
            f for f in self.clean_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ])
        
        self.noisy_images = sorted([
            f for f in self.noisy_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ])
        
        # Verify matching pairs
        if len(self.clean_images) != len(self.noisy_images):
            raise ValueError(
                f"Number of clean ({len(self.clean_images)}) and "
                f"noisy ({len(self.noisy_images)}) images must match"
            )
        
        # Basic transforms
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((image_size, image_size))
        
    def __len__(self):
        return len(self.clean_images)
    
    def __getitem__(self, idx):
        # Load images
        clean_path = self.clean_images[idx]
        noisy_path = self.noisy_images[idx]
        
        clean_img = Image.open(clean_path).convert('RGB')
        noisy_img = Image.open(noisy_path).convert('RGB')
        
        # Resize
        clean_img = self.resize(clean_img)
        noisy_img = self.resize(noisy_img)
        
        # Data augmentation
        if self.augment:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                clean_img = transforms.functional.hflip(clean_img)
                noisy_img = transforms.functional.hflip(noisy_img)
            
            # Random vertical flip
            if torch.rand(1) > 0.5:
                clean_img = transforms.functional.vflip(clean_img)
                noisy_img = transforms.functional.vflip(noisy_img)
            
            # Random rotation (90, 180, 270 degrees)
            if torch.rand(1) > 0.5:
                angle = torch.randint(1, 4, (1,)).item() * 90
                clean_img = transforms.functional.rotate(clean_img, angle)
                noisy_img = transforms.functional.rotate(noisy_img, angle)
        
        # Convert to tensor
        clean_tensor = self.to_tensor(clean_img)
        noisy_tensor = self.to_tensor(noisy_img)
        
        return noisy_tensor, clean_tensor


def create_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    image_size: int = 256,
    train_split: float = 0.9,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Directory containing synthetic dataset
        batch_size: Batch size for training
        image_size: Image size
        train_split: Fraction of data to use for training
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader
    """
    # Create full dataset
    full_dataset = DenoisingDataset(
        data_dir=data_dir,
        image_size=image_size,
        augment=True
    )
    
    # Split into train and validation
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create validation dataset without augmentation
    val_dataset_no_aug = DenoisingDataset(
        data_dir=data_dir,
        image_size=image_size,
        augment=False
    )
    
    # Update val_dataset to use non-augmented version
    val_indices = val_dataset.indices
    val_dataset = torch.utils.data.Subset(val_dataset_no_aug, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate Peak Signal-to-Noise Ratio between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Calculate Structural Similarity Index between two images.
    Simplified version for batch processing.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)
    
    sigma1_sq = torch.var(img1)
    sigma2_sq = torch.var(img2)
    sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(ssim)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str
):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    device: str = 'cuda'
):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
