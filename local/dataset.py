import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import rotate, zoom
import os

class SyntheticMedical3D(Dataset):
    """Generate synthetic 3D medical volumes"""
    def __init__(self, num_samples=1000, volume_size=32, augment=True):
        self.num_samples = num_samples
        self.volume_size = volume_size
        self.augment = augment
        
    def __len__(self):
        return self.num_samples
    
    def create_anatomical_structure(self):
        """Create realistic anatomical structures"""
        volume = np.zeros((self.volume_size, self.volume_size, self.volume_size))
        
        # Random number of structures (organs/tumors)
        num_structures = np.random.randint(2, 5)
        
        for _ in range(num_structures):
            # Random center position
            cx = np.random.randint(8, self.volume_size - 8)
            cy = np.random.randint(8, self.volume_size - 8)
            cz = np.random.randint(8, self.volume_size - 8)
            
            # Random radii (ellipsoid)
            rx = np.random.randint(3, 8)
            ry = np.random.randint(3, 8)
            rz = np.random.randint(3, 8)
            
            # Create 3D grid
            x, y, z = np.ogrid[:self.volume_size, :self.volume_size, :self.volume_size]
            
            # Ellipsoid equation: (x-cx)²/rx² + (y-cy)²/ry² + (z-cz)²/rz² <= 1
            mask = ((x - cx)**2 / rx**2 + 
                    (y - cy)**2 / ry**2 + 
                    (z - cz)**2 / rz**2) <= 1
            
            # Add with varying intensity (tissue density)
            intensity = np.random.uniform(0.6, 1.0)
            volume[mask] = intensity
        
        # Add tissue gradient (simulate CT Hounsfield units variation)
        gradient = np.linspace(0, 0.2, self.volume_size)
        volume += gradient[:, None, None]
        
        # Normalize
        volume = np.clip(volume, 0, 1)
        
        return volume
    
    def augment_volume(self, volume):
        """3D data augmentation"""
        if not self.augment:
            return volume
        
        # Random rotation
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            axis = np.random.randint(0, 3)
            axes = [(1, 2), (0, 2), (0, 1)][axis]
            volume = rotate(volume, angle, axes=axes, reshape=False, order=1)
        
        # Random flip
        if np.random.rand() > 0.5:
            axis = np.random.randint(0, 3)
            volume = np.flip(volume, axis=axis).copy()
        
        # Intensity augmentation
        if np.random.rand() > 0.5:
            volume = volume * np.random.uniform(0.9, 1.1)
            volume = np.clip(volume, 0, 1)
        
        return volume
    
    def __getitem__(self, idx):
        # Create volume
        volume = self.create_anatomical_structure()
        
        # Apply augmentation
        volume = self.augment_volume(volume)
        
        # Add slight Gaussian noise (simulate imaging noise)
        volume += np.random.normal(0, 0.02, volume.shape)
        volume = np.clip(volume, 0, 1)
        
        # Convert to tensor [C, D, H, W]
        volume_tensor = torch.FloatTensor(volume).unsqueeze(0)
        
        return volume_tensor


def save_sample_volumes(save_dir='data/synthetic', num_samples=10):
    """Generate and save sample volumes"""
    os.makedirs(save_dir, exist_ok=True)
    
    dataset = SyntheticMedical3D(num_samples=num_samples, volume_size=32)
    
    for i in range(num_samples):
        volume = dataset[i]
        torch.save(volume, os.path.join(save_dir, f'volume_{i:03d}.pt'))
    
    print(f"✓ Saved {num_samples} synthetic volumes to {save_dir}")