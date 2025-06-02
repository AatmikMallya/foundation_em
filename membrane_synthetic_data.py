import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Tuple, Optional
from torch.utils.data import Dataset, DataLoader
# from scipy.ndimage import gaussian_filter # Not needed for this approach

class MembraneSyntheticDataset(Dataset):
    """Generate 3D volumes with synthetic membrane-like structures (2D sheets in 3D)."""

    def __init__(self,
                 num_samples: int = 1000,
                 volume_size: Tuple[int, int, int] = (64, 64, 64),
                 num_gaussians_range: Tuple[int, int] = (8, 15),
                 gaussian_strength_range: Tuple[float, float] = (-1.0, 1.0),
                 gaussian_sigma_range: Tuple[float, float] = (8.0, 16.0), # Controls feature size
                 isovalue_center: float = 0.0, # For scalar field normalized around 0
                 membrane_band_width: float = 0.1, # Thickness of the membrane band
                 noise_level: Optional[float] = 0.05,
                 normalize_final_volume: bool = True,
                 seed: Optional[int] = None):
        self.num_samples = num_samples
        self.volume_size = volume_size
        self.num_gaussians_range = num_gaussians_range
        self.gaussian_strength_range = gaussian_strength_range
        self.gaussian_sigma_range = gaussian_sigma_range
        self.isovalue_center = isovalue_center
        self.membrane_band_width = membrane_band_width
        self.noise_level = noise_level
        self.normalize_final_volume = normalize_final_volume
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            random.seed(self.seed)

        self.volumes = self._generate_all_volumes()

    def _generate_all_volumes(self):
        volumes = []
        for i in range(self.num_samples):
            current_seed = i + (self.seed if self.seed is not None else np.random.randint(0, 10000))
            np.random.seed(current_seed)
            random.seed(current_seed)
            volumes.append(self._generate_single_volume())
        return volumes

    def _generate_single_volume(self) -> np.ndarray:
        scalar_field = np.zeros(self.volume_size, dtype=np.float32)
        num_gaussians = np.random.randint(self.num_gaussians_range[0], self.num_gaussians_range[1] + 1)

        coords = np.indices(self.volume_size, dtype=np.float32)

        for _ in range(num_gaussians):
            strength = np.random.uniform(self.gaussian_strength_range[0], self.gaussian_strength_range[1])
            sigma_x = np.random.uniform(self.gaussian_sigma_range[0], self.gaussian_sigma_range[1])
            sigma_y = np.random.uniform(self.gaussian_sigma_range[0], self.gaussian_sigma_range[1])
            sigma_z = np.random.uniform(self.gaussian_sigma_range[0], self.gaussian_sigma_range[1])
            
            center_d = np.random.uniform(0, self.volume_size[0])
            center_h = np.random.uniform(0, self.volume_size[1])
            center_w = np.random.uniform(0, self.volume_size[2])

            g = strength * np.exp(-
                (((coords[0] - center_d)**2) / (2 * sigma_z**2)) -
                (((coords[1] - center_h)**2) / (2 * sigma_y**2)) -
                (((coords[2] - center_w)**2) / (2 * sigma_x**2))
            )
            scalar_field += g

        # Normalize scalar field to be roughly centered around 0 with std dev 1 (or just range normalize)
        if np.std(scalar_field) > 1e-6:
             scalar_field = (scalar_field - np.mean(scalar_field)) / np.std(scalar_field)
        
        # Create membrane mask from isoband
        lower_bound = self.isovalue_center - self.membrane_band_width / 2
        upper_bound = self.isovalue_center + self.membrane_band_width / 2
        membrane_mask = (scalar_field >= lower_bound) & (scalar_field <= upper_bound)
        
        volume = membrane_mask.astype(np.float32)

        if self.noise_level is not None and self.noise_level > 0:
            volume += np.random.normal(0, self.noise_level, self.volume_size)
            volume = np.clip(volume, 0.0, 1.0) # Clip after adding noise

        if self.normalize_final_volume:
            # This normalization might not be strictly necessary if mask is 0 or 1 and noise is small
            # but good for consistency if noise is large or future non-binary membranes.
            vol_min, vol_max = volume.min(), volume.max()
            if vol_max > vol_min + 1e-6:
                volume = (volume - vol_min) / (vol_max - vol_min)
        
        return np.clip(volume, 0.0, 1.0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        volume_np = self.volumes[idx]
        return torch.from_numpy(volume_np.copy()).unsqueeze(0) # Add channel dim

    def visualize_sample(self, idx: int = 0, slice_axis: int = 0, slice_idx: Optional[int] = None):
        volume_np = self.volumes[idx]
        if slice_idx is None:
            slice_idx = volume_np.shape[slice_axis] // 2

        slice_2d = np.take(volume_np, slice_idx, axis=slice_axis)
        title = f"Membrane Sample {idx}, Slice {('Z','Y','X')[slice_axis]}={slice_idx}, Range: [{volume_np.min():.2f}-{volume_np.max():.2f}]"
        
        plt.figure(figsize=(8, 8))
        plt.imshow(slice_2d, cmap='gray', vmin=0, vmax=1)
        plt.title(title)
        plt.colorbar()
        plt.show()

def create_membrane_dataloader(batch_size: int = 4,
                               num_samples: int = 1000,
                               volume_size: Tuple[int, int, int] = (64, 64, 64),
                               num_gaussians_range: Tuple[int, int] = (8, 15),
                               gaussian_strength_range: Tuple[float, float] = (-1.0, 1.0),
                               gaussian_sigma_range: Tuple[float, float] = (8.0, 16.0),
                               isovalue_center: float = 0.0,
                               membrane_band_width: float = 0.1,
                               noise_level: Optional[float] = 0.05,
                               num_workers: int = 0,
                               shuffle: bool = True,
                               seed: Optional[int] = None) -> DataLoader:
    dataset = MembraneSyntheticDataset(
        num_samples=num_samples, volume_size=volume_size,
        num_gaussians_range=num_gaussians_range,
        gaussian_strength_range=gaussian_strength_range,
        gaussian_sigma_range=gaussian_sigma_range,
        isovalue_center=isovalue_center,
        membrane_band_width=membrane_band_width,
        noise_level=noise_level,
        normalize_final_volume=True, seed=seed
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

if __name__ == '__main__':
    print("Generating synthetic membrane dataset (sheet-like structures)...")
    dataset = MembraneSyntheticDataset(
        num_samples=5,
        volume_size=(64, 64, 64),
        num_gaussians_range=(10, 20),      # More gaussians for complexity
        gaussian_strength_range=(-1.0, 1.0),
        gaussian_sigma_range=(10.0, 20.0), # Larger sigmas for broader features
        isovalue_center=0.0,               # Center of the band in normalized scalar field
        membrane_band_width=0.15,          # Relatively thin membrane band
        noise_level=0.02,                  # Slight noise on final image
        seed=42
    )
    print(f"Dataset generated with {len(dataset)} samples.")

    if len(dataset) > 0:
        print(f"Visualizing sample 0...")
        try:
            dataset.visualize_sample(idx=0, slice_axis=0) # Z-slice
            dataset.visualize_sample(idx=0, slice_axis=1) # Y-slice
            dataset.visualize_sample(idx=0, slice_axis=2) # X-slice
            
            if len(dataset) > 1:
                 print(f"\nVisualizing sample 1...")
                 dataset.visualize_sample(idx=1, slice_axis=0) 
        except Exception as e:
            print(f"Matplotlib visualization failed: {e}")
    
    print("\nTesting DataLoader...")
    dataloader = create_membrane_dataloader(
        batch_size=2, num_samples=4, volume_size=(32,32,32), # Smaller for quick test
        membrane_band_width=0.2, seed=43, num_gaussians_range=(5,10)
    )
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}: shape {batch.shape}, range [{batch.min():.3f}, {batch.max():.3f}], dtype {batch.dtype}")
        if batch_idx >= 1: break
    
    print("Synthetic membrane data generation (sheets) test completed!") 