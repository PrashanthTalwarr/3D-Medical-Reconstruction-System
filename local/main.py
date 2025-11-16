import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
from datetime import datetime

from local.dataset import SyntheticMedical3D, save_sample_volumes
from local.model import UNet3D
from local.diffusion import DDPM3D
from local.utils import (
    evaluate_reconstruction, visualize_reconstruction_comparison,
    plot_training_curves, plot_metrics_comparison,
    visualize_diffusion_process, EMA, save_checkpoint,
    print_model_summary, count_parameters
)


def train_epoch(model, diffusion, dataloader, optimizer, scaler, device, epoch):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)
        
        # Random timesteps
        t = torch.randint(0, diffusion.timesteps, (batch.shape[0],), device=device).long()
        
        # Forward diffusion: add noise
        noisy_batch, noise = diffusion.q_sample(batch, t)
        
        # Mixed precision training (CUDA optimization)
        with torch.amp.autocast('cuda'):
            # Predict noise using model
            predicted_noise = model(noisy_batch, t)
            
            # Simple MSE loss
            loss = F.mse_loss(predicted_noise, noise)
        
        # Backpropagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = epoch_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def evaluate(model, diffusion, test_loader, device):
    """Evaluate model on test set (UPDATED)"""
    model.eval()
    
    # Get test sample
    test_batch = next(iter(test_loader)).to(device)
    
    # Add noise at middle timestep for visualization
    t_mid = torch.full((test_batch.shape[0],), diffusion.timesteps // 2, device=device).long()
    noisy_batch, _ = diffusion.q_sample(test_batch, t_mid)
    
    # Reconstruct using full diffusion sampling
    print("Generating reconstruction (may take a moment)...")
    reconstructed = diffusion.sample(model, test_batch.shape, device)
    
    # Normalize to [0, 1] range
    reconstructed = torch.clamp(reconstructed, 0, 1)
    test_batch = torch.clamp(test_batch, 0, 1)
    noisy_batch = torch.clamp(noisy_batch, 0, 1)
    
    # Calculate metrics
    metrics = evaluate_reconstruction(reconstructed, test_batch)
    
    # Optionally get intermediate steps for visualization
    intermediate_steps = []
    
    return test_batch, noisy_batch, reconstructed, intermediate_steps, metrics


def main():
    """Main training pipeline"""
    
    print("\n" + "="*60)
    print("3D MEDICAL RECONSTRUCTION SYSTEM")
    print("="*60)
    print("Technologies:")
    print("  ✓ PyTorch - Deep Learning Framework")
    print("  ✓ Diffusion Models (DDPM)")
    print("  ✓ 3D Vision - Volumetric Processing")
    print("  ✓ CUDA - GPU Acceleration")
    print("  ⏳ AWS SageMaker - (Next: sagemaker scripts)")
    print("="*60 + "\n")
    
    # Configuration
    config = {
        'num_samples': 1000,
        'volume_size': 32,
        'batch_size': 4,
        'num_epochs': 30,
        'learning_rate': 1e-4,
        'timesteps': 200,
        'base_channels': 32,
        'save_interval': 10,
    }
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('data/synthetic', exist_ok=True)
    
    # Save configuration
    with open('results/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Generate sample data
    print("Generating synthetic medical data...")
    save_sample_volumes(num_samples=10)
    
    # Dataset and DataLoader
    print(f"\nCreating dataset with {config['num_samples']} samples...")
    train_dataset = SyntheticMedical3D(
        num_samples=config['num_samples'],
        volume_size=config['volume_size'],
        augment=True
    )
    test_dataset = SyntheticMedical3D(
        num_samples=100,
        volume_size=config['volume_size'],
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    print(f"✓ Batch size: {config['batch_size']}")
    
    # Initialize model (PyTorch + 3D Vision)
    print("\nInitializing 3D U-Net model...")
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        base_channels=config['base_channels'],
        time_emb_dim=128
    ).to(device)
    
    print_model_summary(model)
    
    # Initialize diffusion process
    diffusion = DDPM3D(
        timesteps=config['timesteps'],
        device=device
    )
    print(f"✓ Diffusion timesteps: {config['timesteps']}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs']
    )
    
    # Mixed precision scaler (CUDA optimization)
    scaler = torch.amp.GradScaler('cuda')
    
    # EMA for better model stability
    ema = EMA(model, decay=0.9999)
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    
    start_time = datetime.now()
    losses = []
    
    for epoch in range(1, config['num_epochs'] + 1):
        # Train
        avg_loss = train_epoch(
            model, diffusion, train_loader,
            optimizer, scaler, device, epoch
        )
        
        losses.append(avg_loss)
        
        # Update EMA
        ema.update()
        
        # Update learning rate
        scheduler.step()
        
        print(f"Epoch {epoch}/{config['num_epochs']} - Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if epoch % config['save_interval'] == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss)
    
    training_time = datetime.now() - start_time
    print(f"\n✓ Training completed in {training_time}")
    
    # Save final model
    torch.save(model.state_dict(), 'checkpoints/final_model.pt')
    print("✓ Final model saved: checkpoints/final_model.pt")
    
    # Plot training curves
    plot_training_curves(losses)
    
    # Evaluation
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60 + "\n")
    
    original, noisy, reconstructed, diffusion_steps, metrics = evaluate(
        model, diffusion, test_loader, device
    )
    
    # Print metrics
    print("Reconstruction Metrics:")
    print("-" * 40)
    for metric, value in metrics.items():
        emoji = "✅" if (metric == "Dice" and value > 0.85) or (metric == "Accuracy" and value > 90) else "📊"
        print(f"  {emoji} {metric:12s}: {value:.4f}")
    print("-" * 40)
    
    # Visualizations
    print("\nGenerating visualizations...")
    visualize_reconstruction_comparison(original, noisy, reconstructed)
    plot_metrics_comparison(metrics)
    if diffusion_steps:
        visualize_diffusion_process(diffusion_steps)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Model trained for {config['num_epochs']} epochs")
    print(f"✓ Final loss: {losses[-1]:.4f}")
    print(f"✓ Reconstruction accuracy: {metrics['Accuracy']:.2f}%")
    print(f"✓ Dice coefficient: {metrics['Dice']:.4f} (Target: 0.92)")
    print(f"✓ PSNR: {metrics['PSNR']:.2f} dB")
    print(f"✓ SSIM: {metrics['SSIM']:.4f}")
    print(f"✓ Training time: {training_time}")
    print("\nOutput files:")
    print("  - checkpoints/final_model.pt")
    print("  - results/training_loss.png")
    print("  - results/reconstruction_comparison.png")
    print("  - results/metrics_comparison.png")
    print("  - results/diffusion_process.png")
    print("  - results/config.json")
    print("\n" + "="*60)
    print("LOCAL TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review results in 'results/' folder")
    print("  2. Run 'python sagemaker_deploy.py' for AWS deployment")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()