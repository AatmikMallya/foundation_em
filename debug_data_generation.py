import matplotlib.pyplot as plt
import torch
from membrane_synthetic_data import create_membrane_dataloader
import numpy as np

# --- Parameters for Data Generation ---
# Using parameters similar to your training script to see a representative sample.
IMG_SIZE = 64
NUM_SPHERES_RANGE = (10, 20)
RADIUS_RANGE = (10, 20)
NOISE_LEVEL = 0.01
MEMBRANE_BAND_WIDTH = 0.1
# --- Key parameters to verify "added spheres" ---
NUM_ADDED_SPHERES_RANGE = (5, 10) # Using a high number to ensure they are visible
ADDED_SPHERE_RADII_RANGE = (3.0, 6.0)


def visualize_generated_sample():
    """
    Generates and visualizes a single sample from the dataset
    to verify the data generation process.
    """
    print("Generating a single data sample with 'added spheres'...")

    # Use the dataloader to create one sample
    # We set batch_size=1 and num_samples=1
    dataloader = create_membrane_dataloader(
        batch_size=1,
        num_samples=1,
        volume_size=(IMG_SIZE, IMG_SIZE, IMG_SIZE),
        num_gaussians_range=NUM_SPHERES_RANGE,
        gaussian_sigma_range=RADIUS_RANGE,
        noise_level=NOISE_LEVEL,
        membrane_band_width=MEMBRANE_BAND_WIDTH,
        num_additional_spheres_range=NUM_ADDED_SPHERES_RANGE,
        additional_sphere_radius_range=ADDED_SPHERE_RADII_RANGE,
        num_workers=0,
        shuffle=False, # No need to shuffle for one sample
        seed=42 # Use a fixed seed for reproducibility
    )

    # Get the single sample from the dataloader
    # The dataloader yields a batch, so we take the first item
    sample_volume = next(iter(dataloader))

    # The shape will be (B, C, D, H, W), e.g., (1, 1, 64, 64, 64)
    # We remove the batch and channel dimensions for visualization
    volume_np = sample_volume.squeeze().numpy()

    print(f"Generated volume shape: {volume_np.shape}")
    print(f"Data range: min={volume_np.min():.2f}, max={volume_np.max():.2f}")

    # --- Create Visualization ---
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"Slices of Generated Volume with Added Spheres\n"
                 f"Added Spheres: {NUM_ADDED_SPHERES_RANGE}, Radii: {ADDED_SPHERE_RADII_RANGE}",
                 fontsize=16)

    # Get 9 evenly spaced slice indices
    slice_indices = np.linspace(0, volume_np.shape[0] - 1, 9, dtype=int)

    for i, slice_idx in enumerate(slice_indices):
        ax = axes[i // 3, i % 3]
        ax.imshow(volume_np[slice_idx, :, :], cmap='gray')
        ax.set_title(f'Slice Z={slice_idx}')
        ax.axis('off')

    output_filename = "debug_generated_data_with_spheres.png"
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_filename)
    plt.close(fig)

    print(f"\nVisualization saved to: {output_filename}")
    print("Please check the image to confirm the presence of the main membrane and smaller solid spheres.")

if __name__ == '__main__':
    visualize_generated_sample() 