import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime

# ============================================================================
# METRICS
# ============================================================================

def calculate_psnr(pred, target, max_val=1.0):
    """Peak Signal-to-Noise Ratio"""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr


def calculate_ssim_3d(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    """3D Structural Similarity Index"""
    # Simplified 3D SSIM
    mu1 = F.avg_pool3d(pred, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool3d(target, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool3d(pred * pred, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool3d(target * target, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool3d(pred * target, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def calculate_dice_coefficient(pred, target, threshold=0.5):
    """Dice coefficient for anatomical accuracy"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()
    
    if union == 0:
        return 1.0
    
    dice = (2.0 * intersection) / union
    return dice.item()


def calculate_mae(pred, target):
    """Mean Absolute Error"""
    return F.l1_loss(pred, target).item()


def evaluate_reconstruction(pred, target):
    """Comprehensive evaluation metrics"""
    metrics = {
        'MSE': F.mse_loss(pred, target).item(),
        'MAE': calculate_mae(pred, target),
        'PSNR': calculate_psnr(pred, target),
        'SSIM': calculate_ssim_3d(pred, target),
        'Dice': calculate_dice_coefficient(pred, target),
        'Accuracy': ((pred > 0.5) == (target > 0.5)).float().mean().item() * 100
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
    axes[0].imshow(volume[mid_d, :, :], cmap=cmap)
    axes[0].set_title(f'{title} - Axial (Z={mid_d})')
    axes[0].axis('off')
    
    # Coronal slice (front view)
    axes[1].imshow(volume[:, mid_h, :], cmap=cmap)
    axes[1].set_title(f'{title} - Coronal (Y={mid_h})')
    axes[1].axis('off')
    
    # Sagittal slice (side view)
    axes[2].imshow(volume[:, :, mid_w], cmap=cmap)
    axes[2].set_title(f'{title} - Sagittal (X={mid_w})')
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
        
        d, h, w = vol.shape
        mid_d, mid_h, mid_w = d // 2, h // 2, w // 2
        
        # Axial
        axes[i, 0].imshow(vol[mid_d, :, :], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'{title} - Axial')
        axes[i, 0].axis('off')
        
        # Coronal
        axes[i, 1].imshow(vol[:, mid_h, :], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'{title} - Coronal')
        axes[i, 1].axis('off')
        
        # Sagittal
        axes[i, 2].imshow(vol[:, :, mid_w], cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f'{title} - Sagittal')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'reconstruction_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison: {save_path}")
    plt.close()


def plot_training_curves(losses, save_dir='results'):
    """Plot training loss curves"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses, linewidth=2, color='#2E86AB')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    save_path = os.path.join(save_dir, 'training_loss.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved training curve: {save_path}")
    plt.close()


def plot_metrics_comparison(metrics_dict, save_dir='results'):
    """Plot metrics comparison bar chart"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics_names = list(metrics_dict.keys())
    metrics_values = list(metrics_dict.values())
    
    bars = ax.bar(metrics_names, metrics_values, color='#A23B72', alpha=0.8)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Reconstruction Metrics', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'metrics_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved metrics: {save_path}")
    plt.close()


def visualize_diffusion_process(images, save_dir='results'):
    """Visualize the denoising process over timesteps"""
    os.makedirs(save_dir, exist_ok=True)
    
    num_steps = len(images)
    fig, axes = plt.subplots(2, (num_steps + 1) // 2, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, img in enumerate(images):
        img_np = img.cpu().numpy()
        if img_np.ndim == 5:
            img_np = img_np[0, 0, img_np.shape[2]//2, :, :]
        
        axes[idx].imshow(img_np, cmap='gray', vmin=0, vmax=1)
        axes[idx].set_title(f'Step {idx}')
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