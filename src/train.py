"""
Training pipeline for Swin-Unet denoising model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Optional
import math

try:
    from .models.swin_unet import SwinUnet
    from .utils.data_utils import (
        create_dataloaders,
        calculate_psnr,
        save_checkpoint,
        load_checkpoint
    )
except ImportError:
    from models.swin_unet import SwinUnet
    from utils.data_utils import (
        create_dataloaders,
        calculate_psnr,
        save_checkpoint,
        load_checkpoint
    )


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss for image denoising"""
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return torch.mean(loss)


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, channel: int = 3, sigma: float = 1.5) -> torch.Tensor:
    """Differentiable SSIM loss (1-SSIM)."""
    # Create Gaussian window
    coords = torch.arange(window_size, device=pred.device) - window_size // 2
    grid = torch.stack(torch.meshgrid(coords, coords, indexing='ij'), dim=-1).float()
    gaussian = torch.exp(-(grid[..., 0] ** 2 + grid[..., 1] ** 2) / (2 * sigma * sigma))
    gaussian = gaussian / gaussian.sum()
    window = gaussian.expand(channel, 1, window_size, window_size)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return torch.clamp(1 - ssim_map.mean(), 0, 2)


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Encourage edge fidelity using Laplacian filters."""
    laplace_kernel = torch.tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    laplace_kernel = laplace_kernel.repeat(pred.shape[1], 1, 1, 1)

    pred_edges = F.conv2d(pred, laplace_kernel, padding=1, groups=pred.shape[1])
    target_edges = F.conv2d(target, laplace_kernel, padding=1, groups=target.shape[1])
    return F.l1_loss(pred_edges, target_edges)


class DenoisingTrainer:
    """Trainer for Swin-Unet denoising model"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        output_dir: str = 'output/models',
        log_dir: str = 'logs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss functions
        self.criterion_l1 = nn.L1Loss()
        self.criterion_char = CharbonnierLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,
            eta_min=1e-6
        )
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_psnr = 0.0
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_psnr = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (noisy, clean) in enumerate(pbar):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(noisy)
            
            # Calculate losses (content + structure + edges)
            loss_l1 = self.criterion_l1(output, clean)
            loss_char = self.criterion_char(output, clean)
            loss_ssim = ssim_loss(output, clean)
            loss_edges = edge_loss(output, clean)
            loss = 0.30 * loss_l1 + 0.30 * loss_char + 0.25 * loss_ssim + 0.15 * loss_edges
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Calculate PSNR
            with torch.no_grad():
                psnr = calculate_psnr(output, clean)
            
            total_loss += loss.item()
            total_psnr += psnr
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'psnr': f'{psnr:.2f}'
            })
            
            # Log to tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/PSNR', psnr, global_step)
            self.writer.add_scalar('Train/SSIM_Loss', loss_ssim.item(), global_step)
            self.writer.add_scalar('Train/Edge_Loss', loss_edges.item(), global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        avg_psnr = total_psnr / len(self.train_loader)
        
        return avg_loss, avg_psnr
    
    def validate(self, epoch: int):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_psnr = 0
        
        with torch.no_grad():
            for noisy, clean in tqdm(self.val_loader, desc='Validation'):
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                
                # Forward pass
                output = self.model(noisy)
                
                # Calculate loss
                loss_l1 = self.criterion_l1(output, clean)
                loss_char = self.criterion_char(output, clean)
                loss_ssim = ssim_loss(output, clean)
                loss_edges = edge_loss(output, clean)
                loss = 0.30 * loss_l1 + 0.30 * loss_char + 0.25 * loss_ssim + 0.15 * loss_edges
                
                # Calculate PSNR
                psnr = calculate_psnr(output, clean)
                
                total_loss += loss.item()
                total_psnr += psnr
        
        avg_loss = total_loss / len(self.val_loader)
        avg_psnr = total_psnr / len(self.val_loader)
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/PSNR', avg_psnr, epoch)
        self.writer.add_scalar('Val/SSIM_Loss', loss_ssim.item(), epoch)
        self.writer.add_scalar('Val/Edge_Loss', loss_edges.item(), epoch)
        
        return avg_loss, avg_psnr
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """Main training loop"""
        
        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming from checkpoint: {resume_from}")
            self.start_epoch, _ = load_checkpoint(
                self.model,
                self.optimizer,
                resume_from,
                self.device
            )
            self.start_epoch += 1
        
        print(f"Training on device: {self.device}")
        print(f"Number of model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_psnr = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.2f} dB")
            
            # Validate
            val_loss, val_psnr = self.validate(epoch)
            print(f"Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f} dB")
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning Rate: {current_lr:.6f}")
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Save checkpoint
            checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                val_loss,
                str(checkpoint_path)
            )
            
            # Save best model
            if val_psnr > self.best_psnr:
                self.best_psnr = val_psnr
                self.best_val_loss = val_loss
                best_path = self.output_dir / 'best_model.pth'
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    str(best_path)
                )
                print(f"✓ Saved best model with PSNR: {val_psnr:.2f} dB")
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation PSNR: {self.best_psnr:.2f} dB")
        print(f"Best model saved to: {self.output_dir / 'best_model.pth'}")
        print("=" * 60)
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Swin-Unet for image denoising')
    parser.add_argument('--data_dir', type=str, default='output/synthetic',
                        help='Directory with synthetic dataset')
    parser.add_argument('--output_dir', type=str, default='output/models',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for tensorboard logs')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--window_size', type=int, default=8,
                        help='Swin window size (use 8 when image_size is 256)')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    if device == 'cpu' and args.device == 'cuda':
        print("CUDA not available, using CPU")
    
    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        train_split=0.9,
        num_workers=args.num_workers
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = SwinUnet(
        img_size=args.image_size,
        patch_size=4,
        in_chans=3,
        out_chans=3,
        embed_dim=96,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=args.window_size
    )
    
    # Create trainer
    trainer = DenoisingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        log_dir=args.log_dir
    )
    
    # Train
    trainer.train(num_epochs=args.epochs, resume_from=args.resume)


if __name__ == "__main__":
    main()
