import numpy as np
from skimage import measure, morphology
from scipy import ndimage
import time
from numba import jit

def box_count(arr, box_size):
    reshaped = arr.reshape(arr.shape[0] // box_size, box_size,
                           arr.shape[1] // box_size, box_size,
                           arr.shape[2] // box_size, box_size)
    return np.sum(np.any(reshaped, axis=(1, 3, 5)))



def calculate_morph_features(gray_subvol):   
    cytosol_subvol = gray_subvol
    mean = np.mean(cytosol_subvol)
    median = np.median(cytosol_subvol)
    std_dev = np.std(cytosol_subvol)


    features = {
        'cytosol_mean': mean,
        'cytosol_median': median,
        'cytosol_std_dev': std_dev,
    }

    return features
    
# def calculate_morph_features(final_seg, gray_subvol, skel_subvol, mito_subvol):    
    # region = measure.regionprops(final_seg.astype(int))[0]  # only one region, take the first

    # # Basic features
    # volume = region.area
    # bbox = region.bbox
    # centroid = region.centroid
    # equivalent_diameter = region.equivalent_diameter

    # # generate the surface mesh using marching cubes
    # verts, faces, _, _ = measure.marching_cubes(final_seg, level=0.5)
    # surface_area = measure.mesh_surface_area(verts, faces)

    # bbox_volume = (bbox[3] - bbox[0]) * (bbox[4] - bbox[1]) * (bbox[5] - bbox[2])
    # compactness = volume / bbox_volume
    
    # sphericity = (np.pi ** (1/3)) * ((6 * volume) ** (2/3)) / surface_area
    
    # inertia_tensor = region.inertia_tensor
    # eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)
    # eccentricity = np.sqrt(1 - min(eigenvalues) / max(eigenvalues))

    # skeleton = morphology.skeletonize_3d(final_seg)
    # skeleton_length = np.sum(skeleton)

    # distance_transform = ndimage.distance_transform_edt(final_seg)
    # max_thickness = np.max(distance_transform)
    
    # euler_number = measure.euler_number(final_seg)

    # # roughness
    # z = verts[:, 2]  # Extract the z coordinates
    # z_mean = np.mean(z)
    # rms_roughness = np.sqrt(np.mean((z - z_mean) ** 2))

    # # Fractal Dimension (Box Counting method)
    # min_dim = min(final_seg.shape)
    # scales = np.floor(np.logspace(0, np.log2(min_dim) / np.log2(10), num=20, base=2)).astype(int)
    # scales = scales[(scales > 0) & (min_dim % scales == 0)]
    # counts = [box_count(final_seg, scale) for scale in scales]
    # fractal_dim, _ = np.polyfit(np.log(scales), np.log(counts), 1)
    # fractal_dim = -fractal_dim

    # cytosol density statistics
    cytosol_seg = ~final_seg & skel_subvol
    cytosol_subvol = gray_subvol[cytosol_seg]
    mean = np.mean(cytosol_subvol)
    median = np.median(cytosol_subvol)
    std_dev = np.std(cytosol_subvol)

    # # Mitochondrial proximity
    # mito_seg_in_cell = mito_subvol & skel_subvol
    # mito_presence = int(mito_seg_in_cell.any())
    
    features = {
        # 'volume': volume,
        # 'surface_area': surface_area,
        # 'compactness': compactness,
        # 'sphericity': sphericity,
        # 'eccentricity': eccentricity,
        # 'skeleton_length': skeleton_length,
        # 'max_thickness': max_thickness,
        # 'euler_number': euler_number,
        # 'roughness': rms_roughness,
        # 'fractal_dim': fractal_dim,
        'cytosol_mean': mean,
        'cytosol_median': median,
        'cytosol_std_dev': std_dev,
        # 'mito_presence': mito_presence
    }

    return features

def calculate_volume(final_seg):
    region = measure.regionprops(final_seg.astype(int))[0]  # only one region, take the first
    volume = region.area
    return volume
    

def calculate_morph_features_benchmarked(final_seg):
    features = {}
    timings = {}

    def time_feature(feature_name, func, *args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        timings[feature_name] = end_time - start_time
        return result

    # Basic features
    region = time_feature('regionprops', measure.regionprops, final_seg.astype(int))[0]
    features['volume'] = time_feature('volume', lambda: region.area)
    features['bbox'] = time_feature('bbox', lambda: region.bbox)
    features['centroid'] = time_feature('centroid', lambda: region.centroid)
    features['equivalent_diameter'] = time_feature('equivalent_diameter', lambda: region.equivalent_diameter)

    # Surface mesh using marching cubes
    verts, faces, _, _ = time_feature('marching_cubes', measure.marching_cubes, final_seg, level=0.5)
    features['surface_area'] = time_feature('surface_area', measure.mesh_surface_area, verts, faces)

    bbox = features['bbox']
    bbox_volume = (bbox[3] - bbox[0]) * (bbox[4] - bbox[1]) * (bbox[5] - bbox[2])
    features['compactness'] = time_feature('compactness', lambda: features['volume'] / bbox_volume)

    features['sphericity'] = time_feature('sphericity', lambda: (np.pi ** (1/3)) * ((6 * features['volume']) ** (2/3)) / features['surface_area'])

    inertia_tensor = time_feature('inertia_tensor', lambda: region.inertia_tensor)
    eigenvalues, _ = time_feature('eigen_decomposition', np.linalg.eigh, inertia_tensor)
    features['eccentricity'] = time_feature('eccentricity', lambda: np.sqrt(1 - min(eigenvalues) / max(eigenvalues)))

    skeleton = time_feature('skeletonize', morphology.skeletonize_3d, final_seg)
    features['skeleton_length'] = time_feature('skeleton_length', np.sum, skeleton)

    distance_transform = time_feature('distance_transform', ndimage.distance_transform_edt, final_seg)
    features['max_thickness'] = time_feature('max_thickness', np.max, distance_transform)

    features['euler_number'] = time_feature('euler_number', measure.euler_number, final_seg)

    # Roughness measures
    z = verts[:, 2]
    z_mean = np.mean(z)
    features['rms_roughness'] = time_feature('rms_roughness', lambda: np.sqrt(np.mean((z - z_mean) ** 2)))

    projected_area = (bbox[3] - bbox[0]) * (bbox[4] - bbox[1])
    features['surface_area_ratio'] = time_feature('surface_area_ratio', lambda: features['surface_area'] / projected_area)

    # Fractal Dimension (Box Counting method)
    def box_count(arr, box_size):
        reshaped = arr.reshape(arr.shape[0] // box_size, box_size,
                               arr.shape[1] // box_size, box_size,
                               arr.shape[2] // box_size, box_size)
        return np.sum(np.any(reshaped, axis=(1, 3, 5)))
    
    def calculate_fractal_dim():
        min_dim = min(final_seg.shape)
        scales = np.floor(np.logspace(0, np.log2(min_dim) / np.log2(10), num=20, base=2)).astype(int)
        scales = scales[(scales > 0) & (min_dim % scales == 0)]
        counts = [box_count(final_seg, scale) for scale in scales]
        fractal_dim, _ = np.polyfit(np.log(scales), np.log(counts), 1)
        return -fractal_dim

    features['fractal_dimension'] = time_feature('fractal_dimension', calculate_fractal_dim)

    return features, timings

# Example usage:
# features, timings = calculate_morph_features_benchmarked(your_segmentation)
# for feature, time_taken in timings.items():
#     print(f"{feature}: {time_taken:.6f} seconds")
