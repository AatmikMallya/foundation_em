import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import gaussian_filter
import time
import threading

# Global profiler instance for data generation timing
_data_generation_profiler = None
_profiler_lock = threading.Lock()

def set_data_generation_profiler(profiler):
    """Set the global profiler instance for data generation timing."""
    global _data_generation_profiler
    with _profiler_lock:
        _data_generation_profiler = profiler

class MembraneSyntheticDataset(Dataset):
    def __init__(self, 
                 volume_size=(64, 64, 64), 
                 num_gaussians_range=(5, 15), 
                 gaussian_sigma_range=(5, 15), 
                 isovalue=0.5, 
                 isoband_width=0.1, 
                 noise_level=0.05, 
                 num_samples=1000, 
                 seed=42,
                 # --- New parameters for additional spheres ---
                 num_additional_spheres_range=(0, 0),
                 additional_sphere_radius_range=(0, 0),
                 # --- New parameters for improved realism ---
                 blur_sigma=0.5,
                 isovalue_variation=0.1,
                 intensity_gradient_strength=0.3):
        """
        Generates 3D synthetic membrane-like structures on-the-fly with improved realism.

        Args:
            volume_size (tuple): Size of the 3D volume (depth, height, width).
            num_gaussians_range (tuple): (min, max) number of Gaussians to sum.
            gaussian_sigma_range (tuple): (min, max) sigma for Gaussians.
            isovalue (float): Central value for the isoband.
            isoband_width (float): Width of the isoband (isovalue +/- isoband_width/2).
            noise_level (float): Standard deviation of Gaussian noise to add.
            num_samples (int): Number of samples to generate per epoch.
            seed (int): Base random seed for reproducibility.
            num_additional_spheres_range (tuple): (min, max) number of small solid spheres to add.
            additional_sphere_radius_range (tuple): (min, max) radius for the small solid spheres.
            blur_sigma (float): Gaussian blur sigma for softer edges.
            isovalue_variation (float): Range for varying isovalue per sample.
            intensity_gradient_strength (float): Strength of intensity gradients within membranes.
        """
        self.volume_size = volume_size
        self.num_gaussians_range = num_gaussians_range
        self.gaussian_sigma_range = gaussian_sigma_range
        self.isovalue = isovalue
        self.isoband_width = isoband_width
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0  # Track current epoch for seed variation
        self.num_additional_spheres_range = num_additional_spheres_range
        self.additional_sphere_radius_range = additional_sphere_radius_range
        # New parameters for improved realism
        self.blur_sigma = blur_sigma
        self.isovalue_variation = isovalue_variation
        self.intensity_gradient_strength = intensity_gradient_strength

    def _generate_single_sample(self, index, rng_instance):
        """Generates a single 3D volume with a membrane-like structure."""
        global _data_generation_profiler
        current_rng = rng_instance

        D, H, W = self.volume_size
        
        # Initialize scalar field and parameters
        scalar_field = np.zeros((D, H, W), dtype=np.float32)
        num_gaussians = current_rng.randint(self.num_gaussians_range[0], self.num_gaussians_range[1] + 1)
        
        # Create coordinate grids once (optimization)
        d_coords, h_coords, w_coords = np.ogrid[:D, :H, :W]
        
        # Pre-compute constants for optimization
        sigma_min, sigma_max = self.gaussian_sigma_range
        
        # Single gaussian generation function (eliminates code duplication)
        def generate_gaussians():
            nonlocal scalar_field  # Allow modification of outer scope variable
            for _ in range(num_gaussians):
                # Generate all parameters for this Gaussian
                center_d = current_rng.uniform(0, D)
                center_h = current_rng.uniform(0, H) 
                center_w = current_rng.uniform(0, W)
                sigma_d = current_rng.uniform(sigma_min, sigma_max)
                sigma_h = current_rng.uniform(sigma_min, sigma_max)
                sigma_w = current_rng.uniform(sigma_min, sigma_max)
                amplitude = current_rng.uniform(0.5, 1.5)
                
                # Pre-compute inverse variances (optimization)
                inv_2_sigma_d_sq = 1.0 / (2 * sigma_d * sigma_d)
                inv_2_sigma_h_sq = 1.0 / (2 * sigma_h * sigma_h)
                inv_2_sigma_w_sq = 1.0 / (2 * sigma_w * sigma_w)
                
                # Anisotropic Gaussian with optimized computation
                gaussian = amplitude * np.exp(-(
                    ((d_coords - center_d)**2 * inv_2_sigma_d_sq) +
                    ((h_coords - center_h)**2 * inv_2_sigma_h_sq) +
                    ((w_coords - center_w)**2 * inv_2_sigma_w_sq)
                ))
                scalar_field += gaussian
        
        # Profile gaussian field generation with single code path
        if _data_generation_profiler is not None:
            with _data_generation_profiler.profile_section("gaussian_field_generation"):
                generate_gaussians()
        else:
            generate_gaussians()

        # Normalize scalar field to [0, 1] range - optimized single pass
        field_min = np.min(scalar_field)
        field_max = np.max(scalar_field)
        field_range = field_max - field_min
        if field_range > 0:
            scalar_field -= field_min
            scalar_field /= field_range
        else:
            scalar_field.fill(0) # Avoid division by zero if field is flat

        # Vary isovalue per sample for more diversity
        sample_isovalue = self.isovalue + current_rng.uniform(-self.isovalue_variation, self.isovalue_variation)
        sample_isovalue = np.clip(sample_isovalue, 0.1, 0.9)  # Keep within reasonable bounds

        # Define membrane as an isoband with varied isovalue
        lower_bound = sample_isovalue - self.isoband_width / 2
        upper_bound = sample_isovalue + self.isoband_width / 2
        membrane = np.logical_and(scalar_field >= lower_bound, scalar_field <= upper_bound).astype(np.float32)

        # Add intensity gradients within the membrane for more realism
        if self.intensity_gradient_strength > 0:
            # Create a gradient based on distance from membrane center (optimized)
            half_isoband = self.isoband_width * 0.5
            membrane_distance = np.abs(scalar_field - sample_isovalue) / half_isoband
            np.clip(membrane_distance, 0, 1, out=membrane_distance)  # In-place clipping
            # Apply gradient: stronger intensity at membrane center (in-place)
            membrane_distance *= self.intensity_gradient_strength
            membrane_distance = 1.0 - membrane_distance
            membrane *= membrane_distance

        # --- Add additional small spheres ---
        if self.num_additional_spheres_range[1] > 0 and self.additional_sphere_radius_range[1] > 0:
            num_spheres = current_rng.randint(self.num_additional_spheres_range[0], self.num_additional_spheres_range[1] + 1)
            # Reuse existing coordinate grids (optimization - no redundant creation)
            sphere_r_min, sphere_r_max = self.additional_sphere_radius_range
            for _ in range(num_spheres):
                center_d = current_rng.uniform(0, D)
                center_h = current_rng.uniform(0, H)
                center_w = current_rng.uniform(0, W)
                radius = current_rng.uniform(sphere_r_min, sphere_r_max)
                sphere_intensity = current_rng.uniform(0.7, 1.0)
                
                # Create sphere mask with optimized computation (pre-compute radius squared)
                radius_sq = radius * radius
                sphere_mask = ((d_coords - center_d)**2 + (h_coords - center_h)**2 + (w_coords - center_w)**2) < radius_sq
                membrane[sphere_mask] = np.maximum(membrane[sphere_mask], sphere_intensity)

        # Apply Gaussian blur for softer, more realistic edges
        if self.blur_sigma > 0:
            if _data_generation_profiler is not None:
                with _data_generation_profiler.profile_section("gaussian_blur"):
                    membrane = gaussian_filter(membrane, sigma=self.blur_sigma)
            else:
                membrane = gaussian_filter(membrane, sigma=self.blur_sigma)

        # Add noise after blurring (optimized)
        if self.noise_level > 0:
            noise = current_rng.normal(0, self.noise_level, size=self.volume_size).astype(np.float32)
            membrane += noise

        # Add background baseline for more realistic intensity distribution
        # Real EM data rarely has pure black backgrounds (in-place operation)
        membrane += 0.2

        # Final normalization and clipping - in-place for memory efficiency
        np.clip(membrane, 0.0, 1.0, out=membrane)
        
        # Reshape to (1, D, H, W) for channel dimension
        return torch.from_numpy(membrane).unsqueeze(0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        global _data_generation_profiler
        
        # Start timing for data generation
        start_time = time.time()
        
        # Generate seed based on epoch and index to ensure different data each epoch
        seed = self.seed + self.epoch * self.num_samples + idx
        rng = np.random.RandomState(seed)
        
        # Profile the actual data generation
        if _data_generation_profiler is not None:
            with _data_generation_profiler.profile_section("cpu_data_generation"):
                result = self._generate_single_sample(idx, rng_instance=rng)
        else:
            result = self._generate_single_sample(idx, rng_instance=rng)
        
        # Track total data generation time
        generation_time = time.time() - start_time
        if _data_generation_profiler is not None:
            _data_generation_profiler.add_data_generation_time(generation_time)
        
        return result

    def set_epoch(self, epoch):
        """ 
        Sets the current epoch. This is crucial for generating different data
        each epoch when using on-the-fly generation.
        """
        self.epoch = epoch

from torch.utils.data import DataLoader

def create_membrane_dataloader(batch_size, num_samples, volume_size, 
                               num_gaussians_range, gaussian_sigma_range, 
                               noise_level, membrane_band_width, 
                               num_workers, shuffle, seed, 
                               drop_last=True,
                               # Kwargs for additional features
                               **kwargs):
    """
    Creates a DataLoader for the MembraneSyntheticDataset with on-the-fly generation.
    """
    # Extract sphere arguments from kwargs, with defaults
    num_additional_spheres_range = kwargs.get('num_additional_spheres_range', (0,0))
    additional_sphere_radius_range = kwargs.get('additional_sphere_radius_range', (0,0))
    
    # Extract new realism parameters from kwargs, with defaults
    blur_sigma = kwargs.get('blur_sigma', 0.5)
    isovalue_variation = kwargs.get('isovalue_variation', 0.1)
    intensity_gradient_strength = kwargs.get('intensity_gradient_strength', 0.3)

    dataset = MembraneSyntheticDataset(
        volume_size=volume_size,
        num_gaussians_range=num_gaussians_range,
        gaussian_sigma_range=gaussian_sigma_range,
        isoband_width=membrane_band_width, # Mapping argument name
        noise_level=noise_level,
        num_samples=num_samples,
        seed=seed,
        num_additional_spheres_range=num_additional_spheres_range,
        additional_sphere_radius_range=additional_sphere_radius_range,
        blur_sigma=blur_sigma,
        isovalue_variation=isovalue_variation,
        intensity_gradient_strength=intensity_gradient_strength
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True, # Recommended for GPU training
        drop_last=drop_last,   # Use parameter instead of hardcoded True
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
        prefetch_factor=4 if num_workers > 0 else None,  # Each worker prefetches 4 batches
    )
    return dataloader

if __name__ == '__main__':
    # Example usage:
    train_loader_example = create_membrane_dataloader(
        batch_size=2,
        num_samples=10,
        volume_size=(32,32,32),
        num_gaussians_range=(3,8),
        gaussian_sigma_range=(3,10),
        noise_level=0.01,
        membrane_band_width=0.2,
        num_workers=0,
        shuffle=True,
        seed=42,
        # Example of passing args
        num_additional_spheres_range=(2, 5), 
        additional_sphere_radius_range=(2.0, 4.0)
    )

    print(f"Created DataLoader. Number of batches: {len(train_loader_example)}")
    first_batch = next(iter(train_loader_example))
    print(f"First batch shape: {first_batch.shape}")
    print(f"First batch data type: {first_batch.dtype}")