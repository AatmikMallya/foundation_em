import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_dilation, distance_transform_edt
import random
from typing import Tuple, List, Optional
from torch.utils.data import Dataset, DataLoader

class SyntheticEMDataset(Dataset):
    """Generate synthetic 3D volumes that mimic EM microscopy data."""
    
    def __init__(self, 
                 num_samples: int = 1000,
                 volume_size: Tuple[int, int, int] = (64, 64, 64),
                 add_noise: bool = True,
                 normalize: bool = True,
                 seed: Optional[int] = None):
        """
        Args:
            num_samples: Number of synthetic volumes to generate
            volume_size: Size of each volume (D, H, W)
            add_noise: Whether to add realistic noise
            normalize: Whether to normalize intensities to [0, 1]
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.volume_size = volume_size
        self.add_noise = add_noise
        self.normalize = normalize
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
        
        # Pre-generate all volumes for consistent dataset
        self.volumes = self._generate_all_volumes()
    
    def _generate_all_volumes(self):
        """Pre-generate all synthetic volumes."""
        volumes = []
        for i in range(self.num_samples):
            # Set different seed for each volume for variety
            np.random.seed(i + 42)
            volume = self._generate_single_volume()
            volumes.append(volume)
        return volumes
    
    def _generate_single_volume(self) -> np.ndarray:
        """Generate a single synthetic 3D volume with EM-like structures."""
        d, h, w = self.volume_size
        volume = np.zeros((d, h, w), dtype=np.float32)
        
        # 1. Add background texture (simulating cytoplasm)
        background = self._generate_background_texture()
        volume += background
        
        # 2. Add mitochondria-like structures
        volume = self._add_mitochondria(volume)
        
        # 3. Add microtubule-like structures
        volume = self._add_microtubules(volume)
        
        # 4. Add vesicle-like structures
        volume = self._add_vesicles(volume)
        
        # 5. Add membrane-like structures
        volume = self._add_membranes(volume)
        
        # 6. Add realistic noise
        if self.add_noise:
            volume = self._add_noise(volume)
        
        # 7. Normalize
        if self.normalize:
            volume = self._normalize_volume(volume)
        
        return volume
    
    def _generate_background_texture(self) -> np.ndarray:
        """Generate background texture that simulates cytoplasm."""
        d, h, w = self.volume_size
        
        # Create smooth background with some texture
        background = np.random.normal(0.3, 0.1, (d, h, w))
        background = gaussian_filter(background, sigma=2.0)
        
        # Add some graininess
        grain = np.random.normal(0, 0.05, (d, h, w))
        background += grain
        
        return background.astype(np.float32)
    
    def _add_mitochondria(self, volume: np.ndarray) -> np.ndarray:
        """Add mitochondria-like structures (elongated, high contrast)."""
        d, h, w = self.volume_size
        
        # Number of mitochondria
        num_mito = np.random.randint(3, 8)
        
        for _ in range(num_mito):
            # Random center
            center_d = np.random.randint(5, d-5)
            center_h = np.random.randint(5, h-5)
            center_w = np.random.randint(5, w-5)
            
            # Random size (elongated)
            size_d = np.random.randint(8, 20)
            size_h = np.random.randint(4, 10)
            size_w = np.random.randint(4, 10)
            
            # Create ellipsoid
            dd, hh, ww = np.meshgrid(
                np.arange(d) - center_d,
                np.arange(h) - center_h,
                np.arange(w) - center_w,
                indexing='ij'
            )
            
            # Ellipsoid equation
            ellipsoid = (dd**2 / (size_d/2)**2 + 
                        hh**2 / (size_h/2)**2 + 
                        ww**2 / (size_w/2)**2) <= 1
            
            # Add double membrane structure
            outer_membrane = ellipsoid
            inner_scale = 0.7
            inner_ellipsoid = (dd**2 / (size_d*inner_scale/2)**2 + 
                              hh**2 / (size_h*inner_scale/2)**2 + 
                              ww**2 / (size_w*inner_scale/2)**2) <= 1
            
            # High intensity for membranes, lower for interior
            volume[outer_membrane & ~inner_ellipsoid] += 0.8  # Membrane
            volume[inner_ellipsoid] += 0.4  # Interior
        
        return volume
    
    def _add_microtubules(self, volume: np.ndarray) -> np.ndarray:
        """Add microtubule-like structures (thin, tubular)."""
        d, h, w = self.volume_size
        
        # Number of microtubules
        num_mt = np.random.randint(5, 15)
        
        for _ in range(num_mt):
            # Random start and end points
            start = np.array([
                np.random.randint(0, d),
                np.random.randint(0, h),
                np.random.randint(0, w)
            ])
            
            # Create path (roughly straight with some curvature)
            length = np.random.randint(20, min(d, h, w))
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            
            # Trace the microtubule
            radius = 1.5  # Thin tubular structure
            for i in range(length):
                pos = start + i * direction
                pos = pos.astype(int)
                
                # Add some curvature
                if i % 5 == 0:
                    direction += np.random.randn(3) * 0.1
                    direction = direction / np.linalg.norm(direction)
                
                # Check bounds
                if (pos >= 0).all() and (pos < [d, h, w]).all():
                    # Create small sphere around current position
                    for dd in range(-2, 3):
                        for hh in range(-2, 3):
                            for ww in range(-2, 3):
                                new_pos = pos + [dd, hh, ww]
                                if ((new_pos >= 0).all() and 
                                    (new_pos < [d, h, w]).all() and
                                    np.sqrt(dd**2 + hh**2 + ww**2) <= radius):
                                    volume[new_pos[0], new_pos[1], new_pos[2]] += 0.6
        
        return volume
    
    def _add_vesicles(self, volume: np.ndarray) -> np.ndarray:
        """Add vesicle-like structures (small spheres)."""
        d, h, w = self.volume_size
        
        # Number of vesicles
        num_vesicles = np.random.randint(10, 30)
        
        for _ in range(num_vesicles):
            # Random center
            center = np.array([
                np.random.randint(3, d-3),
                np.random.randint(3, h-3),
                np.random.randint(3, w-3)
            ])
            
            # Random radius
            radius = np.random.uniform(1.5, 4.0)
            
            # Create sphere
            dd, hh, ww = np.meshgrid(
                np.arange(d) - center[0],
                np.arange(h) - center[1],
                np.arange(w) - center[2],
                indexing='ij'
            )
            
            sphere = (dd**2 + hh**2 + ww**2) <= radius**2
            
            # High intensity for membrane, low for interior
            membrane = sphere & ((dd**2 + hh**2 + ww**2) > (radius-0.8)**2)
            interior = sphere & ((dd**2 + hh**2 + ww**2) <= (radius-0.8)**2)
            
            volume[membrane] += 0.7
            volume[interior] += 0.2
        
        return volume
    
    def _add_membranes(self, volume: np.ndarray) -> np.ndarray:
        """Add membrane-like structures (sheet-like)."""
        d, h, w = self.volume_size
        
        # Number of membrane sheets
        num_membranes = np.random.randint(2, 6)
        
        for _ in range(num_membranes):
            # Random orientation
            normal = np.random.randn(3)
            normal = normal / np.linalg.norm(normal)
            
            # Random position along normal
            center_dist = np.random.uniform(-min(d,h,w)/4, min(d,h,w)/4)
            
            # Create coordinate grids
            dd, hh, ww = np.meshgrid(
                np.arange(d) - d/2,
                np.arange(h) - h/2, 
                np.arange(w) - w/2,
                indexing='ij'
            )
            
            # Distance from plane
            plane_dist = np.abs(dd*normal[0] + hh*normal[1] + ww*normal[2] - center_dist)
            
            # Membrane thickness
            membrane = plane_dist <= 1.0
            
            # Add some undulation to make it more realistic
            undulation = 0.3 * np.sin(dd/5) * np.sin(hh/5) * np.sin(ww/5)
            membrane = membrane | (plane_dist <= (1.0 + undulation))
            
            volume[membrane] += 0.5
        
        return volume
    
    def _add_noise(self, volume: np.ndarray) -> np.ndarray:
        """Add realistic noise (Gaussian + Poisson-like)."""
        # Gaussian noise
        gaussian_noise = np.random.normal(0, 0.05, volume.shape)
        volume += gaussian_noise
        
        # Shot noise (simplified Poisson) - ensure non-negative input
        volume_positive = np.maximum(volume, 0)  # Clamp to non-negative
        shot_noise = np.random.poisson(np.clip(volume_positive * 10, 0, 100)) / 10 - volume_positive
        volume += shot_noise * 0.1
        
        return volume
    
    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume to [0, 1] range."""
        vol_min = volume.min()
        vol_max = volume.max()
        if vol_max > vol_min:
            volume = (volume - vol_min) / (vol_max - vol_min)
        return volume
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        volume = self.volumes[idx]
        # Add channel dimension for PyTorch
        volume = torch.from_numpy(volume).unsqueeze(0)  # Shape: (1, D, H, W)
        return volume
    
    def visualize_sample(self, idx: int = 0, slice_axis: int = 0, slice_idx: Optional[int] = None):
        """Visualize a sample from the dataset."""
        volume = self.volumes[idx]
        
        if slice_idx is None:
            slice_idx = volume.shape[slice_axis] // 2
        
        if slice_axis == 0:
            slice_2d = volume[slice_idx, :, :]
            title = f"Sample {idx}, Z-slice {slice_idx}"
        elif slice_axis == 1:
            slice_2d = volume[:, slice_idx, :]
            title = f"Sample {idx}, Y-slice {slice_idx}"
        else:
            slice_2d = volume[:, :, slice_idx]
            title = f"Sample {idx}, X-slice {slice_idx}"
        
        plt.figure(figsize=(8, 8))
        plt.imshow(slice_2d, cmap='gray')
        plt.title(title)
        plt.colorbar()
        plt.show()

def create_synthetic_dataloader(batch_size: int = 4,
                               num_samples: int = 1000,
                               volume_size: Tuple[int, int, int] = (64, 64, 64),
                               num_workers: int = 0,
                               shuffle: bool = True) -> DataLoader:
    """Create a DataLoader for synthetic EM data."""
    dataset = SyntheticEMDataset(
        num_samples=num_samples,
        volume_size=volume_size,
        add_noise=True,
        normalize=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader

if __name__ == "__main__":
    # Test the synthetic data generator
    print("Creating synthetic EM dataset...")
    
    # Create dataset
    dataset = SyntheticEMDataset(
        num_samples=10,
        volume_size=(32, 32, 32),
        add_noise=True,
        normalize=True,
        seed=42
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
    
    # Visualize a sample (if matplotlib is available)
    try:
        dataset.visualize_sample(0, slice_axis=0)
        dataset.visualize_sample(0, slice_axis=1) 
        dataset.visualize_sample(0, slice_axis=2)
    except:
        print("Matplotlib not available for visualization")
    
    # Test dataloader
    print("\nTesting DataLoader...")
    dataloader = create_synthetic_dataloader(
        batch_size=2,
        num_samples=10,
        volume_size=(32, 32, 32)
    )
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}: shape {batch.shape}")
        if batch_idx >= 2:  # Only show first few batches
            break
    
    print("Synthetic data generation completed! ✅") 