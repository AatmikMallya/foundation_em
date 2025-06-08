import torch
from torch.utils.data import Dataset
import numpy as np

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
                 additional_sphere_radius_range=(0, 0)):
        """
        Generates 3D synthetic membrane-like structures on-the-fly.

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

    def _generate_single_sample(self, index, rng_instance):
        """Generates a single 3D volume with a membrane-like structure."""
        current_rng = rng_instance

        D, H, W = self.volume_size
        scalar_field = np.zeros((D, H, W), dtype=np.float32)

        num_gaussians = current_rng.randint(self.num_gaussians_range[0], self.num_gaussians_range[1] + 1)

        for _ in range(num_gaussians):
            center_d = current_rng.uniform(0, D)
            center_h = current_rng.uniform(0, H)
            center_w = current_rng.uniform(0, W)
            sigma_d = current_rng.uniform(self.gaussian_sigma_range[0], self.gaussian_sigma_range[1])
            sigma_h = current_rng.uniform(self.gaussian_sigma_range[0], self.gaussian_sigma_range[1])
            sigma_w = current_rng.uniform(self.gaussian_sigma_range[0], self.gaussian_sigma_range[1])
            amplitude = current_rng.uniform(0.5, 1.5) # Randomize amplitude a bit

            d_coords, h_coords, w_coords = np.ogrid[:D, :H, :W]
            
            # Anisotropic Gaussian
            gaussian = amplitude * np.exp(-(
                ((d_coords - center_d)**2 / (2 * sigma_d**2)) +
                ((h_coords - center_h)**2 / (2 * sigma_h**2)) +
                ((w_coords - center_w)**2 / (2 * sigma_w**2))
            ))
            scalar_field += gaussian

        # Normalize scalar field (e.g., to [0, 1] or mean 0, std 1)
        if np.max(scalar_field) > np.min(scalar_field):
            scalar_field = (scalar_field - np.min(scalar_field)) / (np.max(scalar_field) - np.min(scalar_field))
        else:
            scalar_field.fill(0) # Avoid division by zero if field is flat

        # Define membrane as an isoband
        lower_bound = self.isovalue - self.isoband_width / 2
        upper_bound = self.isovalue + self.isoband_width / 2
        membrane = np.logical_and(scalar_field >= lower_bound, scalar_field <= upper_bound).astype(np.float32)

        # --- Add additional small spheres ---
        if self.num_additional_spheres_range[1] > 0 and self.additional_sphere_radius_range[1] > 0:
            num_spheres = current_rng.randint(self.num_additional_spheres_range[0], self.num_additional_spheres_range[1] + 1)
            d_coords, h_coords, w_coords = np.ogrid[:D, :H, :W] # Re-use coordinates
            for _ in range(num_spheres):
                center_d = current_rng.uniform(0, D)
                center_h = current_rng.uniform(0, H)
                center_w = current_rng.uniform(0, W)
                radius = current_rng.uniform(self.additional_sphere_radius_range[0], self.additional_sphere_radius_range[1])
                
                # Create a solid sphere
                sphere_mask = ((d_coords - center_d)**2 + (h_coords - center_h)**2 + (w_coords - center_w)**2) < radius**2
                membrane[sphere_mask] = 1.0 # Add sphere to the volume

        # Add noise
        if self.noise_level > 0:
            noise = current_rng.normal(0, self.noise_level, size=self.volume_size).astype(np.float32)
            membrane += noise
            membrane = np.clip(membrane, 0, 1) # Keep values in a reasonable range after noise
        
        # Reshape to (1, D, H, W) for channel dimension
        return torch.from_numpy(membrane).unsqueeze(0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate seed based on epoch and index to ensure different data each epoch
        seed = self.seed + self.epoch * self.num_samples + idx
        rng = np.random.RandomState(seed)
        return self._generate_single_sample(idx, rng_instance=rng)

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

    dataset = MembraneSyntheticDataset(
        volume_size=volume_size,
        num_gaussians_range=num_gaussians_range,
        gaussian_sigma_range=gaussian_sigma_range,
        isoband_width=membrane_band_width, # Mapping argument name
        noise_level=noise_level,
        num_samples=num_samples,
        seed=seed,
        num_additional_spheres_range=num_additional_spheres_range,
        additional_sphere_radius_range=additional_sphere_radius_range
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True, # Recommended for GPU training
        drop_last=drop_last   # Use parameter instead of hardcoded True
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