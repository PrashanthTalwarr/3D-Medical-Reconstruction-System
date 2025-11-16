import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# Import local modules (they will be copied to SageMaker)
sys.path.append('/opt/ml/code')
from model import UNet3D
from diffusion import DDPM3D
from dataset import SyntheticMedical3D


def train_epoch(model, diffusion, dataloader, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    
    for batch in dataloader:
        batch = batch.to(device)
        t = torch.randint(0, diffusion.timesteps, (batch.shape[0],), device=device).long()
        
        noisy_batch, noise = diffusion.q_sample(batch, t)
        
        with torch.cuda.amp.autocast():
            predicted_noise = model(noisy_batch, t)
            loss = F.mse_loss(predicted_noise, noise)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def main(args):
    """Main training function for SageMaker"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on SageMaker with device: {device}")
    print(f"Hyperparameters: {vars(args)}")
    
    # Dataset
    print("Creating dataset...")
    train_dataset = SyntheticMedical3D(
        num_samples=args.num_samples,
        volume_size=args.volume_size,
        augment=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    
    # Model
    print("Initializing model...")
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        time_emb_dim=128
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Diffusion
    diffusion = DDPM3D(
        timesteps=args.timesteps,
        device=device
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    
    losses = []
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, diffusion, train_loader, optimizer, scaler, device)
        scheduler.step()
        
        losses.append(avg_loss)
        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(args.model_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    model_path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\nFinal model saved: {model_path}")
    
    # Save training metrics
    metrics_path = os.path.join(args.output_data_dir, 'metrics.json')
    os.makedirs(args.output_data_dir, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump({
            'final_loss': losses[-1],
            'losses': losses,
            'epochs': args.epochs,
        }, f)
    
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--timesteps', type=int, default=200)
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--volume_size', type=int, default=32)
    parser.add_argument('--base_channels', type=int, default=32)
    
    # SageMaker specific parameters
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', './data'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    
    args = parser.parse_args()
    
    main(args)