# ============================================================================
# FILE: local/utils.py (COMPLETE UPDATED VERSION)
# Metrics, Visualization, and Helper Functions
# ============================================================================

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime

# ============================================================================
# METRICS (UPDATED WITH FIXES)
# ============================================================================

def calculate_psnr(pred, target, max_val=1.0):
    """Peak Signal-to-Noise Ratio"""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr


def calculate_ssim_3d(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    """3D Structural Similarity Index (FIXED)"""
    # Ensure same device
    pred = pred.to(target.device)
    
    # Normalize to [0, 1]
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    target = (target - target.min()) / (target.max() - target.min() + 1e-8)
    
    # Apply 3D average pooling with proper padding
    kernel_size = min(window_size, pred.shape[2], pred.shape[3], pred.shape[4])
    padding = kernel_size // 2
    
    mu1 = F.avg_pool3d(pred, kernel_size, stride=1, padding=padding)
    mu2 = F.avg_pool3d(target, kernel_size, stride=1, padding=padding)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool3d(pred * pred, kernel_size, stride=1, padding=padding) - mu1_sq
    sigma2_sq = F.avg_pool3d(target * target, kernel_size, stride=1, padding=padding) - mu2_sq
    sigma12 = F.avg_pool3d(pred * target, kernel_size, stride=1, padding=padding) - mu1_mu2
    
    # Add small epsilon to avoid division by zero
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)
    
    return ssim_map.mean().item()


def calculate_dice_coefficient(pred, target, threshold=0.5):
    """Dice coefficient for anatomical accuracy (FIXED)"""
    # Normalize both to [0, 1] range
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    target = (target - target.min()) / (target.max() - target.min() + 1e-8)
    
    # Binarize
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # Calculate intersection and union
    intersection = (pred_binary * target_binary).sum()
    pred_sum = pred_binary.sum()
    target_sum = target_binary.sum()
    
    # Handle edge cases
    if pred_sum == 0 and target_sum == 0:
        return 1.0  # Both empty, perfect match
    if pred_sum == 0 or target_sum == 0:
        return 0.0  # One empty, one not
    
    # Dice coefficient
    dice = (2.0 * intersection) / (pred_sum + target_sum + 1e-8)
    return dice.item()


def calculate_mae(pred, target):
    """Mean Absolute Error"""
    return F.l1_loss(pred, target).item()


def calculate_hausdorff_distance(pred, target, threshold=0.5):
    """Simplified Hausdorff distance for 3D volumes"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # Get surface points (simplified)
    pred_edges = pred_binary - F.max_pool3d(pred_binary, 3, stride=1, padding=1)
    target_edges = target_binary - F.max_pool3d(target_binary, 3, stride=1, padding=1)
    
    # Count edge points
    pred_edge_count = (pred_edges > 0).sum()
    target_edge_count = (target_edges > 0).sum()
    
    if pred_edge_count == 0 or target_edge_count == 0:
        return 0.0
    
    # Simplified distance metric
    diff = torch.abs(pred_edges - target_edges).sum()
    hausdorff = diff / (pred_edge_count + target_edge_count + 1e-8)
    
    return hausdorff.item()


def evaluate_reconstruction(pred, target):
    """Comprehensive evaluation metrics (UPDATED)"""
    # Ensure both are in [0, 1] range
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)
    
    metrics = {
        'MSE': F.mse_loss(pred, target).item(),
        'MAE': calculate_mae(pred, target),
        'PSNR': calculate_psnr(pred, target),
        'SSIM': calculate_ssim_3d(pred, target),
        'Dice': calculate_dice_coefficient(pred, target),
        'Accuracy': ((pred > 0.5) == (target > 0.5)).float().mean().item() * 100,
        'Hausdorff': calculate_hausdorff_distance(pred, target)
    }
    return metrics


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_3d_slices(volume, title="3D Volume", save_path=None, cmap='gray'):
    """Visualize 3D volume with three orthogonal slices"""
    volume = volume.cpu().numpy()
    if volume.ndim == 5:  # [B, C, D, H, W]
        volume = volume[0, 0]
    elif volume.ndim == 4:  # [C, D, H, W]
        volume = volume[0]
    
    d, h, w = volume.shape
    mid_d, mid_h, mid_w = d // 2, h // 2, w // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Axial slice (top view)
    axes[0].imshow(volume[mid_d, :, :], cmap=cmap, vmin=0, vmax=1)
    axes[0].set_title(f'{title} - Axial (Z={mid_d})', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Coronal slice (front view)
    axes[1].imshow(volume[:, mid_h, :], cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title(f'{title} - Coronal (Y={mid_h})', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Sagittal slice (side view)
    axes[2].imshow(volume[:, :, mid_w], cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title(f'{title} - Sagittal (X={mid_w})', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_reconstruction_comparison(original, noisy, reconstructed, save_dir='results'):
    """Compare original, noisy, and reconstructed volumes"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    volumes = [original, noisy, reconstructed]
    titles = ['Original', 'Noisy Input', 'Reconstructed']
    
    for i, (vol, title) in enumerate(zip(volumes, titles)):
        vol = vol.cpu().numpy()
        if vol.ndim == 5:
            vol = vol[0, 0]
        elif vol.ndim == 4:
            vol = vol[0]
        
        # Clip to [0, 1]
        vol = np.clip(vol, 0, 1)
        
        d, h, w = vol.shape
        mid_d, mid_h, mid_w = d // 2, h // 2, w // 2
        
        # Axial
        axes[i, 0].imshow(vol[mid_d, :, :], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'{title} - Axial', fontsize=11, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Coronal
        axes[i, 1].imshow(vol[:, mid_h, :], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'{title} - Coronal', fontsize=11, fontweight='bold')
        axes[i, 1].axis('off')
        
        # Sagittal
        axes[i, 2].imshow(vol[:, :, mid_w], cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f'{title} - Sagittal', fontsize=11, fontweight='bold')
        axes[i, 2].axis('off')
    
    plt.suptitle('3D Medical Image Reconstruction Comparison', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'reconstruction_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison: {save_path}")
    plt.close()


def plot_training_curves(losses, save_dir='results'):
    """Plot training loss curves"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = list(range(1, len(losses) + 1))
    ax.plot(epochs, losses, linewidth=2.5, color='#2E86AB', marker='o', 
            markersize=4, markevery=max(1, len(losses)//20))
    
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_title('Training Loss Over Time', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_yscale('log')
    
    # Add final loss annotation
    final_loss = losses[-1]
    ax.annotate(f'Final: {final_loss:.4f}', 
                xy=(len(losses), final_loss),
                xytext=(len(losses)*0.7, final_loss*2),
                fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_loss.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved training curve: {save_path}")
    plt.close()


def plot_metrics_comparison(metrics_dict, save_dir='results'):
    """Plot metrics comparison bar chart"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    metrics_names = list(metrics_dict.keys())
    metrics_values = list(metrics_dict.values())
    
    # Color scheme
    colors = ['#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#2E86AB', '#BC4B51', '#8B7E74']
    
    bars = ax.bar(metrics_names, metrics_values, 
                   color=colors[:len(metrics_names)], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Value', fontsize=13, fontweight='bold')
    ax.set_title('Reconstruction Performance Metrics', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add target lines for key metrics
    if 'Dice' in metrics_dict:
        ax.axhline(y=0.92, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target Dice (0.92)')
        ax.legend(fontsize=10)
    
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'metrics_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved metrics: {save_path}")
    plt.close()


def visualize_diffusion_process(images, save_dir='results'):
    """Visualize the denoising process over timesteps"""
    os.makedirs(save_dir, exist_ok=True)
    
    num_steps = len(images)
    if num_steps == 0:
        print("⚠ No intermediate images to visualize")
        return
    
    cols = min(5, num_steps)
    rows = (num_steps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, img in enumerate(images):
        img_np = img.cpu().numpy()
        if img_np.ndim == 5:
            img_np = img_np[0, 0, img_np.shape[2]//2, :, :]
        elif img_np.ndim == 4:
            img_np = img_np[0, img_np.shape[1]//2, :, :]
        
        img_np = np.clip(img_np, 0, 1)
        
        axes[idx].imshow(img_np, cmap='gray', vmin=0, vmax=1)
        axes[idx].set_title(f'Step {idx}', fontsize=10, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(images), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Diffusion Denoising Process', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'diffusion_process.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved diffusion process: {save_path}")
    plt.close()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

class EMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def save_checkpoint(model, optimizer, epoch, loss, save_dir='checkpoints', filename=None):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        filename = f'checkpoint_epoch_{epoch}.pt'
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    save_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, save_path)
    print(f"✓ Checkpoint saved: {save_path}")
    
    return save_path


def load_checkpoint(model, optimizer, checkpoint_path, device='cuda'):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"✓ Loaded checkpoint from epoch {epoch} (loss: {loss:.4f})")
    
    return epoch, loss


def count_parameters(model):
    """Count trainable parameters"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


def print_model_summary(model):
    """Print model architecture summary"""
    total_params = count_parameters(model)
    
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"Total Parameters: {total_params:,}")
    print(f"Model Size: ~{total_params * 4 / (1024**2):.2f} MB (FP32)")
    print("="*60 + "\n")