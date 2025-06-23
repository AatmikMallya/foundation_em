#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from membrane_synthetic_data import MembraneSyntheticDataset
import torch

class VolumeViewer:
    def __init__(self, volume):
        """
        Interactive 3D volume viewer with slice navigation.
        
        Args:
            volume: 4D tensor (1, D, H, W) or 3D array (D, H, W)
        """
        # Convert to numpy and remove channel dimension if present
        if isinstance(volume, torch.Tensor):
            self.volume = volume.squeeze().numpy()
        else:
            self.volume = volume.squeeze()
            
        self.D, self.H, self.W = self.volume.shape
        self.current_axis = 0  # 0=Z, 1=Y, 2=X
        self.current_slice = self.D // 2
        
        # Create the figure and subplots
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.25, left=0.1)
        
        # Initial image display
        self.im = self.ax.imshow(self.get_current_slice(), cmap='gray', vmin=0, vmax=1)
        self.ax.set_title(self.get_title())
        self.ax.axis('off')
        
        # Create slider for slice navigation
        slider_ax = plt.axes([0.1, 0.1, 0.65, 0.03])
        self.slider = Slider(
            slider_ax, 'Slice', 
            0, self.get_max_slice(), 
            valinit=self.current_slice, 
            valfmt='%d'
        )
        self.slider.on_changed(self.update_slice)
        
        # Create radio buttons for axis selection
        radio_ax = plt.axes([0.8, 0.1, 0.15, 0.15])
        self.radio = RadioButtons(radio_ax, ('Z-axis (D)', 'Y-axis (H)', 'X-axis (W)'))
        self.radio.on_clicked(self.update_axis)
        
        # Add some info text
        info_text = f"Volume shape: {self.D} × {self.H} × {self.W}\n"
        info_text += f"Min: {self.volume.min():.3f}, Max: {self.volume.max():.3f}\n"
        info_text += f"Mean: {self.volume.mean():.3f}, Std: {self.volume.std():.3f}"
        
        self.fig.text(0.02, 0.95, info_text, fontsize=10, verticalalignment='top', 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
    def get_current_slice(self):
        """Get the current 2D slice based on axis and slice index."""
        if self.current_axis == 0:  # Z-axis
            return self.volume[self.current_slice, :, :]
        elif self.current_axis == 1:  # Y-axis
            return self.volume[:, self.current_slice, :]
        else:  # X-axis
            return self.volume[:, :, self.current_slice]
    
    def get_max_slice(self):
        """Get maximum slice index for current axis."""
        if self.current_axis == 0:
            return self.D - 1
        elif self.current_axis == 1:
            return self.H - 1
        else:
            return self.W - 1
    
    def get_title(self):
        """Get title for current view."""
        axis_names = ['Z (Depth)', 'Y (Height)', 'X (Width)']
        total_slices = [self.D, self.H, self.W]
        return f"{axis_names[self.current_axis]} - Slice {self.current_slice} / {total_slices[self.current_axis]-1}"
    
    def update_slice(self, val):
        """Update displayed slice when slider moves."""
        self.current_slice = int(self.slider.val)
        self.im.set_array(self.get_current_slice())
        self.ax.set_title(self.get_title())
        self.fig.canvas.draw_idle()
    
    def update_axis(self, label):
        """Update axis when radio button is clicked."""
        axis_map = {'Z-axis (D)': 0, 'Y-axis (H)': 1, 'X-axis (W)': 2}
        self.current_axis = axis_map[label]
        
        # Reset slider range and position
        self.slider.valmax = self.get_max_slice()
        self.slider.ax.set_xlim(0, self.get_max_slice())
        self.current_slice = min(self.current_slice, self.get_max_slice())
        self.slider.set_val(self.current_slice)
        
        # Update display
        self.im.set_array(self.get_current_slice())
        self.ax.set_title(self.get_title())
        self.fig.canvas.draw_idle()
    
    def show(self):
        """Display the interactive viewer."""
        plt.show()
        return self.fig

def create_interactive_viewer():
    """
    Create and display an interactive volume viewer.
    """
    print("Creating MembraneSyntheticDataset with your current parameters...")
    
    # Use the user's current parameters
    dataset = MembraneSyntheticDataset(
        volume_size=(64, 64, 64),
        num_gaussians_range=(4, 6),  # User's current setting
        gaussian_sigma_range=(20, 25),  # User's current setting
        isovalue=0.8,  # User's current setting
        isoband_width=0.1,  # User's current setting
        noise_level=0.02,
        num_samples=1,
        seed=42,
        num_additional_spheres_range=(2, 5),
        additional_sphere_radius_range=(3.0, 8.0),
        blur_sigma=1.5,
        isovalue_variation=0.5,  # User's current setting
        intensity_gradient_strength=0.4
    )
    
    print("Generating volume...")
    volume = dataset[0]
    
    print(f"Volume shape: {volume.shape}")
    print(f"Volume range: {volume.min():.4f} - {volume.max():.4f}")
    
    print("\nCreating interactive viewer...")
    print("Instructions:")
    print("• Use the slider to navigate through slices")
    print("• Use radio buttons to switch between Z/Y/X axes")
    print("• Close the window when done")
    
    viewer = VolumeViewer(volume)
    return viewer.show()

if __name__ == "__main__":
    fig = create_interactive_viewer() 