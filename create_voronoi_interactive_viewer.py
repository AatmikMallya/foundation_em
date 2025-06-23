#!/usr/bin/env python3

import numpy as np
from visualize_voronoi_volume import load_voronoi_volume
from create_interactive_html_viewer import create_html_viewer
import os
import subprocess

def create_voronoi_interactive_viewer():
    """
    Create an interactive HTML viewer for the first Voronoi volume.
    """
    print("Creating interactive viewer for first Voronoi volume...")
    
    shard_path = "/gpfs/radev/scratch/clark_damon/am3833/voronoi_volumes_64/shard_00000.tar"
    temp_dir = "/tmp"
    bin_file = "v00000_00000.bin"
    temp_bin_path = os.path.join(temp_dir, bin_file)
    
    # Extract the first volume if not already extracted
    if not os.path.exists(temp_bin_path):
        print(f"Extracting {bin_file} from shard...")
        result = subprocess.run([
            "tar", "-xf", shard_path, "-C", temp_dir, bin_file
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error extracting: {result.stderr}")
            return None
    
    # Load the volume
    print(f"Loading volume from {temp_bin_path}...")
    volume = load_voronoi_volume(temp_bin_path)
    
    print(f"Loaded volume shape: {volume.shape}")
    print(f"Volume range: {volume.min():.4f} - {volume.max():.4f}")
    print(f"Volume mean: {volume.mean():.4f}")
    
    # Convert to torch tensor format expected by the HTML viewer
    import torch
    volume_tensor = torch.from_numpy(volume).unsqueeze(0)  # Add channel dimension
    
    # Create interactive HTML viewer
    output_file = create_html_viewer(volume_tensor, "images/voronoi_first_volume_interactive_viewer.html")
    print(f"Interactive HTML viewer created: {output_file}")
    
    # Clean up
    if os.path.exists(temp_bin_path):
        os.remove(temp_bin_path)
        print("Cleaned up temporary file")
    
    print("Done!")
    return volume_tensor

if __name__ == "__main__":
    volume = create_voronoi_interactive_viewer() 