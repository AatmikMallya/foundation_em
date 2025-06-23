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

class MembraneSyntheticDatasetCustomIntensities(Dataset):
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
                 intensity_gradient_strength=0.3,
                 # --- NEW: Custom intensity control ---
                 background_intensity=0.7,
                 membrane_intensity=0.25,
                 sphere_intensity=0.05):
        """
        Generates 3D synthetic membrane-like structures with custom intensity control.

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
            background_intensity (float): Target mean intensity for background regions (~0.7).
            membrane_intensity (float): Target mean intensity for membrane regions (~0.25).
            sphere_intensity (float): Target mean intensity for sphere regions (~0.05).
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
        # Realism parameters
        self.blur_sigma = blur_sigma
        self.isovalue_variation = isovalue_variation
        self.intensity_gradient_strength = intensity_gradient_strength
        # NEW: Custom intensity control
        self.background_intensity = background_intensity
        self.membrane_intensity = membrane_intensity
        self.sphere_intensity = sphere_intensity

    def _generate_single_sample(self, index, rng_instance):
        """Generates a single 3D volume with custom intensity levels."""
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
        
        # Single gaussian generation function
        def generate_gaussians():
            nonlocal scalar_field
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
        
        # Profile gaussian field generation
        if _data_generation_profiler is not None:
            with _data_generation_profiler.profile_section("gaussian_field_generation"):
                generate_gaussians()
        else:
            generate_gaussians()

        # Normalize scalar field to [0, 1] range
        field_min = np.min(scalar_field)
        field_max = np.max(scalar_field)
        field_range = field_max - field_min
        if field_range > 0:
            scalar_field -= field_min
            scalar_field /= field_range
        else:
            scalar_field.fill(0)

        # Vary isovalue per sample for more diversity
        sample_isovalue = self.isovalue + current_rng.uniform(-self.isovalue_variation, self.isovalue_variation)
        sample_isovalue = np.clip(sample_isovalue, 0.1, 0.9)

        # Define membrane as an isoband with varied isovalue
        lower_bound = sample_isovalue - self.isoband_width / 2
        upper_bound = sample_isovalue + self.isoband_width / 2
        membrane_mask = np.logical_and(scalar_field >= lower_bound, scalar_field <= upper_bound)

        # Initialize final volume with background intensity
        final_volume = np.full(self.volume_size, self.background_intensity, dtype=np.float32)
        
        # Set membrane regions to membrane intensity
        membrane_intensity_with_gradient = np.full(membrane_mask.shape, self.membrane_intensity, dtype=np.float32)
        
        # Add intensity gradients within the membrane for more realism
        if self.intensity_gradient_strength > 0 and np.any(membrane_mask):
            # Create a gradient based on distance from membrane center
            half_isoband = self.isoband_width * 0.5
            membrane_distance = np.abs(scalar_field - sample_isovalue) / half_isoband
            np.clip(membrane_distance, 0, 1, out=membrane_distance)
            
            # Apply gradient: vary intensity within membrane
            gradient_variation = membrane_distance * self.intensity_gradient_strength * self.membrane_intensity
            membrane_intensity_with_gradient = self.membrane_intensity + gradient_variation
            # Keep within reasonable bounds
            np.clip(membrane_intensity_with_gradient, 0.05, 0.5, out=membrane_intensity_with_gradient)
        
        # Apply membrane intensities
        final_volume[membrane_mask] = membrane_intensity_with_gradient[membrane_mask]

        # Add additional small spheres with sphere intensity
        if self.num_additional_spheres_range[1] > 0 and self.additional_sphere_radius_range[1] > 0:
            num_spheres = current_rng.randint(self.num_additional_spheres_range[0], self.num_additional_spheres_range[1] + 1)
            sphere_r_min, sphere_r_max = self.additional_sphere_radius_range
            
            for _ in range(num_spheres):
                center_d = current_rng.uniform(0, D)
                center_h = current_rng.uniform(0, H)
                center_w = current_rng.uniform(0, W)
                radius = current_rng.uniform(sphere_r_min, sphere_r_max)
                
                # Create sphere mask
                radius_sq = radius * radius
                sphere_mask = ((d_coords - center_d)**2 + (h_coords - center_h)**2 + (w_coords - center_w)**2) < radius_sq
                
                # Set sphere regions to sphere intensity with small variation
                sphere_intensity_varied = self.sphere_intensity + current_rng.uniform(-0.02, 0.02)
                sphere_intensity_varied = np.clip(sphere_intensity_varied, 0.01, 0.15)
                final_volume[sphere_mask] = sphere_intensity_varied

        # Apply Gaussian blur for softer, more realistic edges
        if self.blur_sigma > 0:
            if _data_generation_profiler is not None:
                with _data_generation_profiler.profile_section("gaussian_blur"):
                    final_volume = gaussian_filter(final_volume, sigma=self.blur_sigma)
            else:
                final_volume = gaussian_filter(final_volume, sigma=self.blur_sigma)

        # Add noise after blurring
        if self.noise_level > 0:
            noise = current_rng.normal(0, self.noise_level, size=self.volume_size).astype(np.float32)
            final_volume += noise

        # Final clipping to ensure values stay in reasonable range
        np.clip(final_volume, 0.0, 1.0, out=final_volume)
        
        # Reshape to (1, D, H, W) for channel dimension
        return torch.from_numpy(final_volume).unsqueeze(0)

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

# Convenience function to create the dataset with custom intensities
def create_custom_intensity_membrane_dataset(volume_size=(64, 64, 64),
                                            num_gaussians_range=(4, 6),
                                            gaussian_sigma_range=(20, 25),
                                            background_intensity=0.7,
                                            membrane_intensity=0.25,
                                            sphere_intensity=0.05,
                                            **kwargs):
    """
    Create a membrane dataset with custom intensity levels.
    
    Args:
        background_intensity (float): Mean intensity for background (~0.7 for light background)
        membrane_intensity (float): Mean intensity for membranes (~0.25 for dark membranes)
        sphere_intensity (float): Mean intensity for spheres (~0.05 for very dark spheres)
    """
    return MembraneSyntheticDatasetCustomIntensities(
        volume_size=volume_size,
        num_gaussians_range=num_gaussians_range,
        gaussian_sigma_range=gaussian_sigma_range,
        background_intensity=background_intensity,
        membrane_intensity=membrane_intensity,
        sphere_intensity=sphere_intensity,
        **kwargs
    ) 