# % matplotlib inline
from neuprint import Client, fetch_roi_hierarchy, skeleton
from neuprint import fetch_synapses, NeuronCriteria as NC, SynapseCriteria as SC
from neuprint.queries import fetch_mitochondria
import numpy as np
import pandas as pd
import time
import importlib
import random
from os.path import isfile
from skimage import measure
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, binary_dilation, binary_closing, binary_erosion, measurements, convolve, center_of_mass
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_prominences
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cdist
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import plotly

spec = importlib.util.spec_from_file_location('config', os.path.dirname(__file__) + '/config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
home_dir = '/Users/aatmikmallya/Desktop/research/fly/synapseSegmentation'
def import_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

voxel_utils = import_module('voxel_utils', f'{home_dir}/util_files/voxel_utils.py')


import warnings
warnings.filterwarnings("error") # force warnings to be errors
warnings.filterwarnings("ignore") # ignore all warnings

token_id = config.token_id
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
home_dir = config.home_dir
c = config.c
server = config.server
node_class_dict = config.node_class_dict

# neuron_quality = pd.read_csv(home_dir + '/saved_data/neuron_quality.csv')
# neuron_quality_np = neuron_quality.to_numpy()

# import utils file
spec = importlib.util.spec_from_file_location('utils', home_dir+'/util_files/utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

def post_process_segmentation(mask):
    # struct_element = np.ones((3,3,3))
    mask = binary_closing(mask)    
    return mask

def create_hard_membrane_buffer(cell_mask, buffer_size=5):
    buffer = cell_mask & ndimage.binary_erosion(cell_mask, iterations=buffer_size)
    return buffer

def create_soft_membrane_buffer(cell_mask, max_distance=4):
    distances = ndimage.distance_transform_edt(cell_mask)
    
    normalized_distance = np.clip(distances / max_distance, 0, 1)
    weight = normalized_distance ** 3
    # weight = 1 - np.exp(-normalized_distance)
    
    # Set all points outside the cell to 0
    weight[~cell_mask] = 0
    
    return weight


def segment_synapse(coord, bodyId, neuron_type, i_synapse):
    gray_subvol = voxel_utils.get_subvols_batched(init_boxes_zyx, 'grayscale')
    skel_subvol = voxel_utils.get_subvols_batched(init_boxes_zyx, 'segmentation')
    
    # Calculate init probability matrix for segmentation
    gray_min, gray_max = np.quantile(gray_subvol.flatten(), [0.01, 0.95])
    init_prob = np.interp(gray_subvol, [gray_min, gray_max], [1,0])**2
    # init_prob[~skel_subvol] = 0

    buffer = create_soft_membrane_buffer(skel_subvol)
    init_prob *= buffer
    
    # do relaxation labeling
    final_seg = segmentation.RelaxationLabeling(init_prob)

    # Label connected components
    labels = measure.label(final_seg)

    # Calculate the centers of mass for each labeled segment
    segment_centers = center_of_mass(final_seg, labels, range(1, labels.max() + 1))
    
    # Filter out small segments
    regions = measure.regionprops(labels)
    min_area = 200
    valid_indices = [i for i, region in enumerate(regions) if region.area > min_area]

    if not valid_indices:
        raise Exception(f'No segment larger than {min_area} area')
        
    valid_centers = [segment_centers[i] for i in valid_indices]
    valid_labels = [i + 1 for i in valid_indices]  # Labels start from 1

    # Find the segment whose center of mass is closest to the center
    center_of_subvol = np.array([50, 50, 50])
    distances = np.linalg.norm(np.array(valid_centers) - center_of_subvol, axis=1)
    closest_segment_index = np.argmin(distances)
    closest_segment = valid_labels[closest_segment_index]
    # set all non-background segments to white
    final_seg_rgb = np.zeros((*final_seg.shape, 3), dtype=np.float32)
    final_seg_rgb[...] = final_seg[..., np.newaxis]

    final_seg = labels == closest_segment

    seg_data = {
        'i_synapse': i_synapse,
        'bodyId': bodyId,
        'neuron_type': neuron_type,
        'gray_subvol': gray_subvol,
        'init_prob': init_prob,
        'final_seg_rgb': final_seg_rgb,
        'gray_min': gray_min,
        'gray_max': gray_max,
        'closest_segment': closest_segment,
        'labels': labels
    }
    return final_seg, seg_data

def segment_synapse_batched(gray_subvol, skel_subvol, mito_subvol, coord, bodyId, neuron_type, i_synapse, relax_params):
    # Calculate init probability matrix for segmentation
    gray_min, gray_max = np.quantile(gray_subvol.flatten(), [0.01, 0.95])
    # gray_min, gray_max = np.quantile(gray_subvol.flatten(), [0.01, 0.99])
    init_prob = np.interp(gray_subvol, [gray_min, gray_max], [1,0]) ** 3

    # init_prob[~skel_subvol] = 0
    init_prob = np.where(mito_subvol, 0, init_prob)

    buffer = create_soft_membrane_buffer(skel_subvol)
    init_prob *= buffer
    
    # do relaxation labeling
    final_seg = RelaxationLabeling(init_prob, **relax_params)
    
    # Label connected components
    labels = measure.label(final_seg)

    # Calculate the centers of mass for each labeled segment
    segment_centers = center_of_mass(final_seg, labels, range(1, labels.max() + 1))
    
    # Filter out small segments
    regions = measure.regionprops(labels)
    min_area = 100
    max_area = 10000
    valid_indices = [i for i, region in enumerate(regions) if min_area < region.area < max_area]

    if not valid_indices:
        raise Exception(f'No segments within area range: ({min_area}, {max_area})')
        
    valid_centers = [segment_centers[i] for i in valid_indices]
    valid_labels = [i + 1 for i in valid_indices]  # Labels start from 1

    # Find the segment whose center of mass is closest to the center
    center_of_subvol = np.array([50, 50, 50])
    distances = np.linalg.norm(np.array(valid_centers) - center_of_subvol, axis=1)
    closest_segment_index = np.argmin(distances)
    closest_segment = valid_labels[closest_segment_index]
    # set all non-background segments to white
    final_seg_rgb = np.zeros((*final_seg.shape, 3), dtype=np.float32)
    final_seg_rgb[...] = final_seg[..., np.newaxis]

    final_seg = labels == closest_segment

    seg_data = {
        'i_synapse': i_synapse,
        'bodyId': bodyId,
        'neuron_type': neuron_type,
        'gray_subvol': gray_subvol,
        'init_prob': init_prob,
        'final_seg_rgb': final_seg_rgb,
        'gray_min': gray_min,
        'gray_max': gray_max,
        'closest_segment': closest_segment,
        'labels': labels
    }
    return final_seg, seg_data
    

def RelaxationLabeling(prob_map, num_iters=20, delta=0.5, offset=1, early_stop_threshold=0.05):
    kernel_size = 2*offset + 1
    xv, yv, zv = np.meshgrid(np.arange(kernel_size)-offset,
                         np.arange(kernel_size)-offset,
                         np.arange(kernel_size)-offset)
    
    dists = np.sqrt( xv**2 + yv**2 + zv**2 )
    conv_kernel = np.where( dists <= 1, 1, 0 )
    conv_kernel[offset, offset, offset] = 0 # exclude current voxel
    conv_kernel = conv_kernel / np.sum(conv_kernel)

    for i in range(num_iters):
        old_prob_map = prob_map.copy()
        
         # Support to be label 1
        support = convolve(2 * prob_map - 1, conv_kernel, mode = 'mirror')
        prob_map += delta * support
        prob_map = np.clip(prob_map, 0, 1)

        # Early stopping
        if np.max(np.abs(prob_map - old_prob_map)) < early_stop_threshold:
            # print(f"Early stopping at iteration {i+1}")
            break

    final_mask = prob_map > 0.5
    # final_mask = post_process_segmentation(final_mask)
    
    return final_mask

# def RelaxationLabeling(prob_map, num_iters=50, offset=1, delta=0.5, early_stop_threshold=0.05):
#     num_iters = 10
#     kernel_size = 2*offset + 1
    
#     x, y, z = np.ogrid[-offset:offset+1, -offset:offset+1, -offset:offset+1]
#     kernel = (x*x + y*y + z*z <= 1).astype(float)
#     kernel[offset, offset, offset] = 0
#     kernel /= kernel.sum()
    
#     for i in range(num_iters):
#         old_prob_map = prob_map.copy()
        
#         support = convolve(2 * prob_map - 1, kernel, mode='mirror')
#         prob_map += delta * support
#         np.clip(prob_map, 0, 1, out=prob_map)
        
#         # Early stopping
#         if np.max(np.abs(prob_map - old_prob_map)) < early_stop_threshold:
#             print(f"Early stopping at iteration {i+1}")
#             break
    
#     return prob_map > 0.5


# Plotting functions
def plot_segmentation_2d(seg_data):
    i_synapse = seg_data['i_synapse']
    bodyId = seg_data['bodyId']
    neuron_type = seg_data['neuron_type']
    gray_subvol = seg_data['gray_subvol']
    init_prob = seg_data['init_prob']
    final_seg_rgb = seg_data['final_seg_rgb']
    gray_min = seg_data['gray_min']
    gray_max = seg_data['gray_max']
    closest_segment = seg_data['closest_segment']
    labels = seg_data['labels']

    fig, axes = plt.subplots( figsize=(8,8) , ncols = 3, nrows = 3)

    axes[0,0].imshow(gray_subvol[50,:,:], vmin=gray_min, vmax=gray_max, cmap = 'gray')
    axes[0,1].imshow(gray_subvol[:,50,:], vmin=gray_min, vmax=gray_max, cmap = 'gray')
    axes[0,2].imshow(gray_subvol[:,:,50], vmin=gray_min, vmax=gray_max, cmap = 'gray')

    axes[0,0].set_ylabel('Grayscale Values')

    axes[1,0].imshow(init_prob[50,:,:], vmin=0, vmax=1, cmap = 'gray')
    axes[1,1].imshow(init_prob[:,50,:], vmin=0, vmax=1, cmap = 'gray')
    axes[1,2].imshow(init_prob[:,:,50], vmin=0, vmax=1, cmap = 'gray')

    axes[1,0].set_ylabel('Initial Probability')

    # highlight closest segment (i.e. the synapse)
    final_seg_rgb[labels == closest_segment] = config.section_colors['axon']

    axes[2,0].imshow(final_seg_rgb[50,:,:])
    axes[2,1].imshow(final_seg_rgb[:,50,:])
    axes[2,2].imshow(final_seg_rgb[:,:,50])

    axes[2,0].set_ylabel('Synapse Mask')

    fontsize=12
    axes[0,0].set_title('View 1', fontsize=fontsize)
    axes[0,1].set_title('View 2', fontsize=fontsize)
    axes[0,2].set_title('View 3', fontsize=fontsize)

    plt.show()

    
def plot_synapse_3d(final_seg, seg_data):
    i_synapse = seg_data['i_synapse']
    bodyId = seg_data['bodyId']
    neuron_type = seg_data['neuron_type']
    
    ls = LightSource(azdeg=0, altdeg=20)
    newMin = 0.2
    newMax = 1
    newdiff = newMax-newMin

    verts, faces, normals, values = measure.marching_cubes(final_seg)
    min_coords = np.min(verts,axis=0)
    max_coords = np.max(verts,axis=0)
    mesh = Poly3DCollection(verts[faces])
    normalsarray = np.array([np.array((np.sum(normals[face[:], 0]/3), np.sum(normals[face[:], 1]/3), np.sum(normals[face[:], 2]/3))/np.sqrt(np.sum(normals[face[:], 0]/3)**2 + np.sum(normals[face[:], 1]/3)**2 + np.sum(normals[face[:], 2]/3)**2)) for face in faces])

    # Next this is more asthetic, but it prevents the shadows of the image being too dark. (linear interpolation to correct)
    min_val = np.min(ls.shade_normals(normalsarray, fraction=1.0)) # min shade value
    max_val = np.max(ls.shade_normals(normalsarray, fraction=1.0)) # max shade value
    diff = max_val-min_val

    colourRGB = np.array(config.section_colors['axon'])
    # The correct shading for shadows are now applied. Use the face normals and light orientation to generate a shading value and apply to the RGB colors for each face.
    rgbNew = np.array([colourRGB*(newMin + newdiff*((shade-min_val)/diff)) for shade in ls.shade_normals(normalsarray, fraction=1.0)])
    rgbNew = np.append(rgbNew, np.ones((len(rgbNew),1)) , axis=1)
    mesh.set_facecolor(rgbNew)
    
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(mesh)

    ax.set_xlim(min_coords[0]-1, max_coords[0]+1)  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(min_coords[1]-1, max_coords[1]+1)  # b = 10
    ax.set_zlim(min_coords[2]-1, max_coords[2]+1)  # c = 16

    ax.view_init(90,-90) #elev,azim) 90, -90
    #ax.axis('off')
    ax.set_box_aspect(max_coords - min_coords)

    plt.tight_layout()
    plt.title(f'{bodyId} {neuron_type} {i_synapse}')
    plt.axis('off')
    plt.show()

def plot_synapse_3d_interactive(final_seg, seg_data):
    i_synapse = seg_data['i_synapse']
    bodyId = seg_data['bodyId']
    neuron_type = seg_data['neuron_type']
    
    verts, faces, normals, values = measure.marching_cubes(final_seg)
    min_coords = np.min(verts,axis=0)
    max_coords = np.max(verts,axis=0)
    
    x, y, z = verts.T
    i, j, k = faces.T
    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color=utils.list_to_rgb_string(config.section_colors['axon']),
        lighting=dict(ambient=0.7,diffuse=0.8,specular=0.1,roughness=0.5,fresnel=0.2)
    )
    
    fig = go.Figure(data=[mesh])
    fig.update_layout(
        width=800,
        height=600,
        scene=dict(
            xaxis=dict(nticks=10, range=[min_coords[0]-1, max_coords[0]+1]),
            yaxis=dict(nticks=10, range=[min_coords[1]-1, max_coords[1]+1]),
            zaxis=dict(nticks=10, range=[min_coords[2]-1, max_coords[2]+1]),
            aspectmode='data'
        ),
        title=f'{bodyId} {neuron_type} {i_synapse}',
        margin=dict(r=0, l=0, b=0, t=40)
    )
    fig.show()

# Deprecated
def get_cached_subvoxel(coord, volume_type, bodyId=None):
    # Define the bounding box for the subvolume
    coord_flipped = np.flip(coord)
    box_zyx = [coord_flipped - 50, coord_flipped + 50]
    
    cache_dir = os.path.join('cache_subvol', volume_type)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    coord_str = '_'.join(map(str, coord))
    filename = os.path.join(cache_dir, f"{coord_str}.pkl")
    
    if os.path.exists(filename):
        # print(f'Retrieving cached {volume_type} subvolume, box_zyx = {coord_str}')
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        if volume_type == 'grayscale':
            # print(f'Downloading subvolume, box_zyx = {coord_str}')
            data = voxel_utils.get_subvol_any_size(box_zyx, 'grayscale').astype(np.uint8)
        elif volume_type == 'segmentation':
            data = voxel_utils.get_subvol_any_size(box_zyx, 'segmentation') == bodyId
        else:
            raise ValueError("Invalid volume type. Must be 'grayscale' or 'segmentation'.")
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        return data
