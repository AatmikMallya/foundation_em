import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Tuple, List, Optional
from torch.utils.data import Dataset, DataLoader

class SimpleSyntheticDataset(Dataset):
    """Generate very simple 3D volumes with basic geometric shapes with randomized properties."""
    
    def __init__(self, 
                 num_samples: int = 1000,
                 volume_size: Tuple[int, int, int] = (32, 32, 32),
                 max_shapes_per_volume: int = 3, # Max number of shapes in a single volume
                 add_noise_level: Optional[float] = None, # e.g., 0.02 for small noise
                 normalize: bool = True,
                 seed: Optional[int] = None):
        """
        Args:
            num_samples: Number of synthetic volumes to generate
            volume_size: Size of each volume (D, H, W)
            max_shapes_per_volume: Max number of shapes in a single volume
            add_noise_level: Optional noise level for the dataset
            normalize: Whether to normalize intensities to [0, 1]
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.volume_size = volume_size
        self.max_shapes_per_volume = max_shapes_per_volume
        self.add_noise_level = add_noise_level
        self.normalize = normalize
        self.seed = seed # Store seed as an instance attribute
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
        
        self.volumes = self._generate_all_volumes()
    
    def _generate_all_volumes(self):
        """Pre-generate all synthetic volumes."""
        volumes = []
        for i in range(self.num_samples):
            np.random.seed(i + (self.seed if self.seed is not None else 42)) # Ensure variety but allow overall seed
            volume = self._generate_single_volume()
            volumes.append(volume)
        return volumes
    
    def _generate_single_volume(self) -> np.ndarray:
        """Generate a single simple 3D volume."""
        d, h, w = self.volume_size
        volume = np.zeros((d, h, w), dtype=np.float32)
        
        num_shapes = np.random.randint(1, self.max_shapes_per_volume + 1)
        
        for _ in range(num_shapes):
            shape_type = np.random.choice(['sphere', 'cube', 'cylinder'])
            intensity = np.random.uniform(0.5, 1.0) # Randomized intensity
            
            if shape_type == 'sphere':
                volume = self._add_sphere(volume, intensity)
            elif shape_type == 'cube':
                volume = self._add_cube(volume, intensity)
            elif shape_type == 'cylinder':
                volume = self._add_cylinder(volume, intensity)
        
        if self.add_noise_level is not None and self.add_noise_level > 0:
            volume = self._add_noise(volume, self.add_noise_level)
        
        if self.normalize:
            volume = self._normalize_volume(volume)
        
        # Ensure background is 0 if no shapes were added or all shapes are outside bounds somehow
        # (especially after potential normalization of an all-zero volume)
        if np.all(volume < 1e-3):
             volume = np.zeros((d, h, w), dtype=np.float32)
        elif self.normalize: # Renormalize if clipping changed things significantly
            volume = self._clip_and_renormalize(volume)

        return volume
    
    def _clip_and_renormalize(self, volume: np.ndarray) -> np.ndarray:
        # Clip values to be safe, especially if multiple shapes overlap and exceed 1.0 before normalization
        # or noise pushes values outside [0,1] post-normalization
        volume = np.clip(volume, 0.0, 1.0)
        # Re-normalize if the dataset is intended to be normalized and clipping occurred
        # This ensures that if an object had intensity 0.5 and background 0, it remains so relative to max 1.
        # If max was e.g. 0.5, it would scale it up. We want to preserve [0,1] if already in range.
        if np.max(volume) > 0: # Avoid division by zero for all-black images
             if not (np.min(volume) >= 0 and np.max(volume) <=1 and np.isclose(np.max(volume), 1.0)):
                  # Only re-normalize if not already in a good [0,1] state with max near 1
                  volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-6)
        return np.clip(volume, 0.0, 1.0) # Final clip to be sure
    
    def _add_sphere(self, volume: np.ndarray, intensity: float) -> np.ndarray:
        """Add a single sphere to the volume."""
        d, h, w = self.volume_size
        min_dim = min(d,h,w)
        
        center_d = np.random.randint(min_dim//6, 5*min_dim//6)
        center_h = np.random.randint(min_dim//6, 5*min_dim//6)
        center_w = np.random.randint(min_dim//6, 5*min_dim//6)
        
        max_radius = min_dim // 4 # Max radius relative to smallest dimension
        radius = np.random.randint(min_dim//8, max_radius + 1)
        if radius < 2: radius = 2 # Ensure minimum radius

        dd, hh, ww = np.meshgrid(
            np.arange(d) - center_d, np.arange(h) - center_h, np.arange(w) - center_w, indexing='ij'
        )
        sphere_mask = (dd**2 + hh**2 + ww**2) <= radius**2
        volume[sphere_mask] = np.maximum(volume[sphere_mask], intensity) # Use maximum in case of overlap
        return volume
    
    def _add_cube(self, volume: np.ndarray, intensity: float) -> np.ndarray:
        """Add a single cube to the volume."""
        d, h, w = self.volume_size
        min_dim = min(d,h,w)

        max_edge = min_dim // 2
        edge_size = np.random.randint(min_dim//8, max_edge + 1)
        if edge_size < 3: edge_size = 3

        start_d = np.random.randint(0, d - edge_size + 1)
        start_h = np.random.randint(0, h - edge_size + 1)
        start_w = np.random.randint(0, w - edge_size + 1)
        
        cube_slice = (slice(start_d, start_d + edge_size), 
                      slice(start_h, start_h + edge_size), 
                      slice(start_w, start_w + edge_size))
        volume[cube_slice] = np.maximum(volume[cube_slice], intensity)
        return volume
    
    def _add_cylinder(self, volume: np.ndarray, intensity: float) -> np.ndarray:
        """Add a cylinder along one axis."""
        d, h, w = self.volume_size
        min_dim = min(d,h,w)
        axis = np.random.randint(0, 3)
        
        max_radius = min_dim // 4
        radius = np.random.randint(min_dim//8, max_radius + 1)
        if radius < 2: radius = 2

        cylinder_mask = np.zeros_like(volume, dtype=bool)
        if axis == 0: # Along D
            center_h, center_w = np.random.randint(radius, h-radius+1), np.random.randint(radius, w-radius+1)
            hh, ww = np.meshgrid(np.arange(h) - center_h, np.arange(w) - center_w, indexing='ij')
            circle = (hh**2 + ww**2) <= radius**2
            for i in range(d): cylinder_mask[i, circle] = True
        elif axis == 1: # Along H
            center_d, center_w = np.random.randint(radius, d-radius+1), np.random.randint(radius, w-radius+1)
            dd, ww = np.meshgrid(np.arange(d) - center_d, np.arange(w) - center_w, indexing='ij')
            circle = (dd**2 + ww**2) <= radius**2
            for j in range(h): cylinder_mask[circle, j] = True # dd, ww order is for mask[d_indices, w_indices]
        else: # Along W
            center_d, center_h = np.random.randint(radius, d-radius+1), np.random.randint(radius, h-radius+1)
            dd, hh = np.meshgrid(np.arange(d) - center_d, np.arange(h) - center_h, indexing='ij')
            circle = (dd**2 + hh**2) <= radius**2
            for k in range(w): cylinder_mask[circle, k] = True
        
        volume[cylinder_mask] = np.maximum(volume[cylinder_mask], intensity)
        return volume
    
    def _add_noise(self, volume: np.ndarray, noise_level: float) -> np.ndarray:
        """Add noise to the volume."""
        noise = np.random.normal(0, noise_level, volume.shape)
        volume += noise
        return volume
    
    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume to [0, 1] range."""
        vol_min = volume.min()
        vol_max = volume.max()
        if vol_max > vol_min + 1e-6: # Avoid division by zero if volume is constant
            if vol_max > 1.0 or vol_min < 0.0 or (vol_max < 0.5 and vol_max > 1e-6): # Normalize if out of [0,1] or max is too low
                volume = (volume - vol_min) / (vol_max - vol_min)
        return np.clip(volume, 0.0, 1.0) # Clip to ensure [0,1] range strictly
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        volume = self.volumes[idx]
        return torch.from_numpy(volume.copy()).unsqueeze(0) # Ensure channel dim and copy
    
    def visualize_sample(self, idx: int = 0, slice_axis: int = 0, slice_idx: Optional[int] = None):
        """Visualize a sample from the dataset."""
        volume_np = self.volumes[idx]
        if slice_idx is None: slice_idx = volume_np.shape[slice_axis] // 2
        
        slice_2d = volume_np.take(indices=slice_idx, axis=slice_axis)
        title = f"Sample {idx}, Slice {('Z','Y','X')[slice_axis]}={slice_idx}, Range: [{volume_np.min():.2f}-{volume_np.max():.2f}]"
        
        plt.figure(figsize=(6, 6))
        plt.imshow(slice_2d, cmap='gray', vmin=0, vmax=1)
        plt.title(title); plt.colorbar(); plt.show()

def create_simple_dataloader(batch_size: int = 4,
                            num_samples: int = 1000,
                            volume_size: Tuple[int, int, int] = (32, 32, 32),
                            max_shapes_per_volume: int = 3,
                            add_noise_level: Optional[float] = None,
                            num_workers: int = 0,
                            shuffle: bool = True,
                            seed: Optional[int] = None) -> DataLoader:
    """Create a DataLoader for simple synthetic data."""
    dataset = SimpleSyntheticDataset(
        num_samples=num_samples,
        volume_size=volume_size,
        max_shapes_per_volume=max_shapes_per_volume,
        add_noise_level=add_noise_level,
        normalize=True,
        seed=seed
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
    # Test the simple synthetic data generator
    print("Creating diverse simple synthetic dataset...")
    
    # Create dataset
    dataset = SimpleSyntheticDataset(
        num_samples=10,
        volume_size=(32, 32, 32),
        max_shapes_per_volume=2,
        add_noise_level=0.01,
        seed=42
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"Sample 0: shape {sample.shape}, range [{sample.min():.3f}, {sample.max():.3f}], dtype {sample.dtype}")
    
    # Visualize several samples
    try:
        for i in range(min(3, len(dataset))):
            dataset.visualize_sample(i, slice_axis=0)
            # dataset.visualize_sample(i, slice_axis=1)
    except Exception as e: print(f"Matplotlib visualization failed: {e}")
    
    # Test dataloader
    print("\nTesting DataLoader...")
    dataloader = create_simple_dataloader(
        batch_size=2,
        num_samples=10,
        volume_size=(32, 32, 32),
        max_shapes_per_volume=2,
        add_noise_level=0.01,
        seed=42
    )
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}: shape {batch.shape}, range [{batch.min():.3f}, {batch.max():.3f}]")
        if batch_idx >= 1: break
    
    print("Diverse simple synthetic data generation completed! ✅") 