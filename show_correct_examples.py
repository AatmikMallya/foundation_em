#!/usr/bin/env python3
"""
Show correct examples of all structure combinations in synthetic EM data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Copy the MembraneGen class and modify it to force specific combinations
class MembraneGenForced:
    def __init__(self, generate_masks=False, forced_combo=None):
        # static params baked in; adapt if needed
        self.n_gauss = (2, 4); self.sigma = (35, 45)  # Reduced from (4,6) for simpler membranes
        self.iso, self.band = 0.8, 0.08  # Keep thin membranes
        self.noise = 0.01  # Reduced from 0.02
        
        # IMPROVED: Varied sphere parameters
        self.n_sph = (2, 16)  # Variable count: 4-8 spheres when present
        self.sph_r = (4.0, 12.0)  # Variable radius: 4-12 pixels
        
        # IMPROVED: Varied cube parameters  
        self.n_cube = (2, 16)  # Variable count: 2-6 cubes when present
        self.cube_size = (8.0, 16.0)  # Variable size: 8-16 pixels
        
        self.blur, self.iso_var, self.grad = 0.5, 0.2, 0.2 # Reduced blur and iso_var
        
        # IMPROVED: Base intensities with variation
        self.bg_base = 0.72
        self.mem_base = 0.22
        self.sph_base = 0.03
        self.cube_base = 0.05
        self.intensity_variation = 0.02  # ±2% intensity variation
        
        self.generate_masks = generate_masks
        self.forced_combo = forced_combo  # Force this specific combination

    def check_collision_vectorized(self, placed_centers, placed_radii, new_center, new_radius, min_separation=3.0):
        """Vectorized collision check using NumPy arrays"""
        if len(placed_centers) == 0:
            return False
        
        # Vectorized distance calculation
        new_c = np.array(new_center)
        distances_sq = np.sum((placed_centers - new_c)**2, axis=1)
        required_sep_sq = (placed_radii + new_radius + min_separation)**2
        
        return np.any(distances_sq < required_sep_sq)

    def __call__(self, seed: int):
        D, H, W = 96, 96, 96
        _dg, _hg, _wg = np.ogrid[:D, :H, :W]
        
        rng = np.random.RandomState(int(seed))
        
        # Use forced combination if provided
        if self.forced_combo is not None:
            combo = self.forced_combo
        else:
            # Default fallback
            combo = {'membrane': True, 'spheres': True, 'cubes': True}
        
        # Generate intensity values with variation
        bg_intensity = self.bg_base + rng.uniform(-self.intensity_variation, self.intensity_variation)
        mem_intensity = self.mem_base + rng.uniform(-self.intensity_variation, self.intensity_variation)
        sph_intensity = self.sph_base + rng.uniform(-self.intensity_variation, self.intensity_variation)
        cube_intensity = self.cube_base + rng.uniform(-self.intensity_variation, self.intensity_variation)
        
        # Clamp intensities to reasonable ranges
        bg_intensity = np.clip(bg_intensity, 0.6, 0.8)
        mem_intensity = np.clip(mem_intensity, 0.15, 0.35)
        sph_intensity = np.clip(sph_intensity, 0.01, 0.08)
        cube_intensity = np.clip(cube_intensity, 0.02, 0.1)
        
        field = np.zeros((D, H, W), np.float32)

        # Generate membrane structures only if chosen
        if combo['membrane']:
            for _ in range(rng.randint(*self.n_gauss)+1):
                cd,ch,cw = rng.uniform(0,D), rng.uniform(0,H), rng.uniform(0,W)
                sd,sh,sw = (rng.uniform(*self.sigma) for _ in range(3))
                amp = rng.uniform(0.5,1.5)
                inv = [1/(2*s*s) for s in (sd,sh,sw)]
                field += amp * np.exp(-((_dg-cd)**2*inv[0]+(_hg-ch)**2*inv[1]+(_wg-cw)**2*inv[2]))

        # Initialize volume with background
        vol = np.full((D, H, W), bg_intensity, np.float32)
        
        # Process membrane field only if membranes are chosen
        mem_mask = None
        if combo['membrane']:
            field -= field.min(); mx = field.max()
            if mx>0: field/=mx

            iso = np.clip(self.iso+rng.uniform(-self.iso_var,self.iso_var),0.1,0.9)
            lo,hi = iso-self.band/2, iso+self.band/2
            mem_mask = (field>=lo)&(field<=hi)

            if mem_mask.any():
                dist = np.abs(field-iso)/(self.band*0.5)
                grad = np.clip(dist,0,1)*self.grad*mem_intensity
                mem_vals = np.clip(mem_intensity+grad,0.05,0.5)
                vol[mem_mask]=mem_vals[mem_mask]

        # Initialize segmentation mask if needed
        if self.generate_masks:
            seg_mask = np.zeros((D, H, W), dtype=np.uint8)  # 0 = background
            if combo['membrane'] and mem_mask is not None and mem_mask.any():
                seg_mask[mem_mask] = 1  # 1 = membrane

        # Compute distance transform for membrane collision detection only if membranes exist
        if combo['membrane'] and mem_mask is not None and mem_mask.any():
            from scipy.ndimage import distance_transform_edt
            dist_to_membrane = distance_transform_edt(~mem_mask)
        else:
            dist_to_membrane = np.full((D, H, W), float('inf'), dtype=np.float32)

        # Track placed shapes for collision detection (vectorized)
        placed_centers = []
        placed_radii = []

        # Generate spheres only if chosen
        if combo['spheres']:
            ns0,ns1 = self.n_sph
            target_spheres = rng.randint(ns0, ns1+1)
            placed_spheres = 0
            
            for _ in range(target_spheres):
                cd,ch,cw = rng.uniform(0,D),rng.uniform(0,H),rng.uniform(0,W)
                r = rng.uniform(*self.sph_r)  # Variable radius
                
                # Fast distance-based membrane collision check
                center_coords = (int(np.clip(cd, 0, D-1)), int(np.clip(ch, 0, H-1)), int(np.clip(cw, 0, W-1)))
                min_gap = 2.0
                membrane_clear = dist_to_membrane[center_coords] >= (r + min_gap)
                
                # Vectorized collision check with other organelles
                placed_centers_arr = np.array(placed_centers) if placed_centers else np.empty((0, 3))
                placed_radii_arr = np.array(placed_radii) if placed_radii else np.empty(0)
                organelle_clear = not self.check_collision_vectorized(placed_centers_arr, placed_radii_arr, (cd,ch,cw), r)
                
                if membrane_clear and organelle_clear:
                    sph_mask = ((_dg-cd)**2+(_hg-ch)**2+(_wg-cw)**2)<r*r
                    # Variable intensity per sphere
                    sval = sph_intensity + rng.uniform(-0.01, 0.01)
                    sval = np.clip(sval, 0.01, 0.15)
                    vol[sph_mask]=sval

                    if self.generate_masks:
                        seg_mask[sph_mask] = 2  # 2 = sphere
                    
                    placed_centers.append((cd,ch,cw))
                    placed_radii.append(r)
                    placed_spheres += 1

        # Generate cubes only if chosen
        if combo['cubes']:
            nc0,nc1 = self.n_cube
            target_cubes = rng.randint(nc0, nc1+1)
            placed_cubes = 0
            
            for _ in range(target_cubes):
                cd,ch,cw = rng.uniform(0,D),rng.uniform(0,H),rng.uniform(0,W)
                size = rng.uniform(*self.cube_size)  # Variable size
                
                # Fast distance-based membrane collision check
                center_coords = (int(np.clip(cd, 0, D-1)), int(np.clip(ch, 0, H-1)), int(np.clip(cw, 0, W-1)))
                min_gap = 2.0
                # Use half-diagonal of cube as effective radius for collision check
                effective_radius = size * np.sqrt(3) / 2
                membrane_clear = dist_to_membrane[center_coords] >= (effective_radius + min_gap)
                
                # Vectorized collision check with other organelles
                placed_centers_arr = np.array(placed_centers) if placed_centers else np.empty((0, 3))
                placed_radii_arr = np.array(placed_radii) if placed_radii else np.empty(0)
                organelle_clear = not self.check_collision_vectorized(placed_centers_arr, placed_radii_arr, (cd,ch,cw), effective_radius)
                
                if membrane_clear and organelle_clear:
                    # Create cube mask
                    half_size = size / 2
                    cube_mask = (
                        (np.abs(_dg - cd) <= half_size) &
                        (np.abs(_hg - ch) <= half_size) &
                        (np.abs(_wg - cw) <= half_size)
                    )
                    
                    # Variable intensity per cube
                    cval = cube_intensity + rng.uniform(-0.01, 0.01)
                    cval = np.clip(cval, 0.01, 0.15)
                    vol[cube_mask] = cval

                    if self.generate_masks:
                        seg_mask[cube_mask] = 3  # 3 = cube
                    
                    placed_centers.append((cd, ch, cw))
                    placed_radii.append(effective_radius)
                    placed_cubes += 1

        # Fast separable blur (3x faster than gaussian_filter)
        if self.blur > 0:
            from scipy.ndimage import convolve1d
            # Create 1D Gaussian kernel
            sigma = self.blur
            kernel_size = int(4 * sigma + 1)  # Reasonable kernel size
            if kernel_size % 2 == 0:
                kernel_size += 1
            x = np.arange(kernel_size) - kernel_size // 2
            kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
            kernel_1d /= kernel_1d.sum()
            
            # Apply separable 1D convolutions
            vol = convolve1d(vol, kernel_1d, axis=0, mode='nearest')
            vol = convolve1d(vol, kernel_1d, axis=1, mode='nearest')
            vol = convolve1d(vol, kernel_1d, axis=2, mode='nearest')
        if self.noise>0: vol+=rng.normal(0,self.noise,(D, H, W)).astype(np.float32)
        np.clip(vol,0,1,out=vol)
        
        if self.generate_masks:
            return vol.tobytes(), seg_mask.tobytes()
        else:
            return vol.tobytes()

def generate_combination_examples():
    """Generate examples of each structure combination"""
    print("Generating examples of each combination...")
    
    # Define all 8 combinations with descriptive names
    combinations = [
        {'name': 'Background Only', 'membrane': False, 'spheres': False, 'cubes': False},
        {'name': 'Membranes Only', 'membrane': True, 'spheres': False, 'cubes': False},
        {'name': 'Spheres Only', 'membrane': False, 'spheres': True, 'cubes': False},
        {'name': 'Cubes Only', 'membrane': False, 'spheres': False, 'cubes': True},
        {'name': 'Membranes + Spheres', 'membrane': True, 'spheres': True, 'cubes': False},
        {'name': 'Membranes + Cubes', 'membrane': True, 'spheres': False, 'cubes': True},
        {'name': 'Spheres + Cubes', 'membrane': False, 'spheres': True, 'cubes': True},
        {'name': 'All Structures', 'membrane': True, 'spheres': True, 'cubes': True},
    ]
    
    volumes = []
    masks = []
    combo_info = []
    
    # Generate one example of each combination
    for i, combo in enumerate(combinations):
        print(f"  Generating {combo['name']} (seed {i*100 + 42})")
        
        # Create generator with forced combination
        gen = MembraneGenForced(generate_masks=True, forced_combo=combo)
        
        seed = i * 100 + 42
        vol_bytes, mask_bytes = gen(seed)
        
        volume = np.frombuffer(vol_bytes, dtype=np.float32).reshape(96, 96, 96)
        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(96, 96, 96)
        
        volumes.append(volume)
        masks.append(mask)
        combo_info.append(combo)
        
        # Print stats
        unique_labels = np.unique(mask)
        counts = np.bincount(mask.flatten())
        print(f"    Labels: {unique_labels}, Counts: {counts[:len(unique_labels)]}")
    
    return volumes, masks, combo_info

def create_static_examples():
    """Create static slice examples showing different combinations"""
    print("Creating static slice examples...")
    
    volumes, masks, combo_info = generate_combination_examples()
    
    # Create a large figure showing middle slices of each combination
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, (volume, mask, combo) in enumerate(zip(volumes, masks, combo_info)):
        ax = axes[i]
        
        # Show middle slice (z=48)
        z_slice = 48
        vol_norm = (volume - volume.min()) / (volume.max() - volume.min())
        
        # Show overlay
        ax.imshow(vol_norm[z_slice], cmap='gray', vmin=0, vmax=1, alpha=0.8)
        
        # Overlay shapes with transparency
        membrane_mask = mask[z_slice] == 1
        sphere_mask = mask[z_slice] == 2
        cube_mask = mask[z_slice] == 3
        
        if membrane_mask.any():
            ax.imshow(np.where(membrane_mask, 1, np.nan), cmap='Blues', alpha=0.6, vmin=0, vmax=1)
        if sphere_mask.any():
            ax.imshow(np.where(sphere_mask, 1, np.nan), cmap='Reds', alpha=0.6, vmin=0, vmax=1)
        if cube_mask.any():
            ax.imshow(np.where(cube_mask, 1, np.nan), cmap='Oranges', alpha=0.6, vmin=0, vmax=1)
        
        ax.set_title(f"{combo['name']}\n(Middle slice Z={z_slice})", fontsize=12)
        ax.axis('off')
        
        # Add stats text
        unique_labels = np.unique(mask)
        counts = np.bincount(mask.flatten())
        stats_text = []
        label_names = ['BG', 'Mem', 'Sph', 'Cube']
        for j, count in enumerate(counts[:len(unique_labels)]):
            if j < len(label_names):
                pct = (count / mask.size) * 100
                stats_text.append(f"{label_names[j]}: {pct:.1f}%")
        
        ax.text(0.02, 0.98, '\n'.join(stats_text), transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
    
    plt.suptitle('Synthetic EM Volume Examples: All Structure Combinations\n(Blue=Membrane, Red=Sphere, Orange=Cube)', 
                 fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('correct_structure_combinations.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Correct examples saved as correct_structure_combinations.png")
    return 'correct_structure_combinations.png'

def main():
    """Main function to create example visualizations"""
    print("=== Synthetic EM Data Examples (Corrected) ===")
    
    # Create examples of all combinations
    create_static_examples()
    
    print("\n✓ Complete! Generated corrected visualization file:")
    print("  - correct_structure_combinations.png: All 8 possible combinations (properly forced)")

if __name__ == "__main__":
    main() 