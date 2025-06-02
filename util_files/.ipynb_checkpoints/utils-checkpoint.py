import numpy as np
import pandas as pd
from neuprint import Client, skeleton
import matplotlib.pyplot as plt
import copy
import networkx as nx
import importlib
from scipy import stats
from ast import literal_eval
import os
from os.path import isfile
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from skimage.morphology import binary_closing
from scipy import ndimage
from skimage import exposure, filters, restoration
from skimage import filters, morphology
from skimage.restoration import denoise_nl_means, estimate_sigma
import warnings
warnings.filterwarnings("ignore") # ignore all warnings

def import_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
home_dir = '/Users/aatmikmallya/Desktop/research/fly/segmentation'
voxel_utils = import_module('voxel_utils', f'{home_dir}/util_files/voxel_utils.py')

# import config file
spec = importlib.util.spec_from_file_location('config', os.path.dirname(__file__) + '/config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

token_id = config.token_id
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
home_dir = config.home_dir
c = config.c
server = config.server
# uuid of the hemibrain-flattened repository
node_class_dict = config.node_class_dict

# os.environ['TENSORSTORE_CA_BUNDLE'] = config.tensorstore_ca_bundle
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config.google_application_credentials

# neuron_quality = pd.read_csv(home_dir + '/saved_data/neuron_quality.csv')
# neuron_quality_np = neuron_quality.to_numpy()

def apply_mask(image, mask):
    return np.where(mask, image, 0)
    
def denoise_image(image, mask, weight=0.1, mask_cell=True):
    if mask_cell:
        image = apply_mask(image, mask)
    denoised = restoration.denoise_tv_chambolle(image, weight=weight, channel_axis=None)
    if mask_cell:
        denoised = apply_mask(denoised, mask)
    return denoised

def get_slice_from_box(init_box_zyx, bodyId, mask_cell=True):
    mito_subvol = voxel_utils.get_subvols_batched([init_box_zyx], 'mito-objects') == [bodyId]
    skel_subvol = voxel_utils.get_subvols_batched([init_box_zyx], 'segmentation') == [bodyId]
    gray_subvol = voxel_utils.get_subvols_batched([init_box_zyx], 'grayscale_clahe')

    mito_subvol = mito_subvol[0]
    skel_subvol = skel_subvol[0]
    gray_subvol = gray_subvol[0]

    skel_subvol = binary_closing(skel_subvol)

    if mask_cell:
        gray_subvol = np.where(skel_subvol, gray_subvol, 0)
        # gray_subvol = np.where(mito_subvol, 0, gray_subvol)

    return gray_subvol, skel_subvol, mito_subvol


def spherical_to_cartesian(radius, theta, phi):
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.array([x, y, z])

def create_3d_slices_animation(images, titles, axis='y', interval=120, cmap='gray', viridis_indices=[], cols=None):
    """
    Create an animation of 3D image slices with independent scaling per image.
    
    Parameters:
        images: List of 3D numpy arrays
        titles: List of strings for image titles
        axis: String indicating slice axis ('x', 'y', or 'z')
        interval: Animation interval in milliseconds
        cmap: Default colormap for non-viridis images
        viridis_indices: List of indices where viridis colormap should be used
        cols: Number of columns in the subplot grid
    """
    axis_index = {'x': 0, 'y': 1, 'z': 2}[axis]
    n_images = len(images)
    if cols:
        n_cols = cols
    else:
        n_cols = min(3, n_images)
    n_rows = int(np.ceil(n_images / n_cols))
    n_frames = images[0].shape[axis_index]
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = np.array(axes).flatten()
    
    # Pre-configure axes and create image objects with independent scaling
    img_objects = []
    for i, (ax, img, title) in enumerate(zip(axes[:n_images], images, titles)):
        ax.set_title(title)
        ax.axis('off')
        
        # Set individual vmin/vmax for each image
        vmin = img.min()
        vmax = img.max()
        
        # Initialize with the first slice
        initial_slice = np.take(img, 0, axis=axis_index)
        if i in viridis_indices:
            im_obj = ax.imshow(initial_slice, cmap='viridis', vmin=vmin, vmax=vmax)
        else:
            im_obj = ax.imshow(initial_slice, cmap=cmap, vmin=vmin, vmax=vmax)
        img_objects.append(im_obj)
    
    # Hide unused subplots
    for ax in axes[n_images:]:
        ax.axis('off')
    
    title_obj = fig.suptitle(f'{axis.upper()}-axis Slice 0', fontsize=16)
    
    def update(frame):
        for img, im_obj in zip(images, img_objects):
            slice_img = np.take(img, frame, axis=axis_index)
            im_obj.set_data(slice_img)
        title_obj.set_text(f'{axis.upper()}-axis Slice {frame}')
        return img_objects + [title_obj]
    
    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True)
    plt.close(fig)
    
    return HTML(anim.to_jshtml())

def find_closest_direction(theta, phi):
    # Convert spherical coordinates to cartesian (normal vector)
    normal = np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    # Unit vectors for each axis
    axes = {
        'x': np.array([1, 0, 0]),
        'y': np.array([0, 1, 0]),
        'z': np.array([0, 0, 1])
    }
    # Calculate angles between normal and each axis
    angles = {
        direction: np.arccos(np.abs(np.dot(normal, axis)))
        for direction, axis in axes.items()
    }
    # Find the axis with minimum angle
    direction, angle = min(angles.items(), key=lambda x: x[1])
    return direction, angle

def list_to_rgb_string(rgb_list):
    return f"rgb({rgb_list[0]},{rgb_list[1]},{rgb_list[2]})"

def spherical_2_cart(r,theta,phi):

    # given r, theta, and phi, compute the cartesian coordinates
    x = r*np.cos(phi)*np.sin(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(theta)
    return x, y, z

def cart_2_spherical(x,y,z):

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos( z / r )
    if x==0:
        # if y=0, phi is undefined but we'll call it zero
        phi = 0 if y==0 else np.sign(y) * np.pi/2
    elif y==0:
        # x is non-zero but y is 0
        phi = 0 if x>0 else np.pi
    else:
        # x and y are non-zero, so figure out which quadrant we're in
        if (x>0) and (y>0): phi = np.arctan(y/x)
        elif (x<0) and (y>0): phi = np.arctan(np.abs(x)/y) + np.pi/2
        elif (x<0) and (y<0): phi = np.arctan( np.abs(y) / np.abs(x) ) + np.pi
        elif (x>0) and (y<0):phi = np.arctan( x / np.abs(y) ) + 3*np.pi/2
    return r, theta, phi

def get_synapse_df(bodyId, neuron_type, synapse_type, group_synapses = True):
    '''
    Get the dataframe describing synapses

    '''

    keep_cols = ['type', 'confidence', 'x', 'y', 'z', 'connecting_bodyId', 'connecting_type', 'connecting_x', 'connecting_y', 'connecting_z', 'neuron_type', 'bodyId']

    neuron_ids = ['a','b','d','c'] if synapse_type == 'pre' else ['d', 'c', 'a', 'b']
    rois = c.all_rois

    q = f"""\
        MATCH (a:Neuron)-[:Contains]->(:`SynapseSet`)-[:Contains]->(b:Synapse)-[:SynapsesTo]->(c:Synapse)<-[:Contains]-(:`SynapseSet`)<-[:Contains]-(d:Neuron)
        WHERE {neuron_ids[0]}.bodyId = {bodyId}
        RETURN {neuron_ids[0]}.bodyId as bodyId,
               {neuron_ids[0]}.type as neuron_type,
               {neuron_ids[1]}.confidence as confidence,
               {neuron_ids[1]}.location.x as x,
               {neuron_ids[1]}.location.y as y,
               {neuron_ids[1]}.location.z as z,
               {neuron_ids[2]}.bodyId as connecting_bodyId,
               {neuron_ids[2]}.type as connecting_type,
               {neuron_ids[3]}.location.x as connecting_x,
               {neuron_ids[3]}.location.y as connecting_y,
               {neuron_ids[3]}.location.z as connecting_z
    """


    synapse_sites = c.fetch_custom(q).drop_duplicates()
    synapse_sites['type'] = synapse_type
    synapse_sites['neuron_type'] = neuron_type
    synapse_sites['bodyId'] = bodyId
    if len(synapse_sites) == 0:
        # this skeleton has no presynaptic sites
        return None

    neuron_type = synapse_sites.iloc[0]['neuron_type']

    coords = synapse_sites[['x','y','z']].to_numpy()
    connecting_coords = synapse_sites[['connecting_x','connecting_y','connecting_z']].to_numpy()

    synapse_ids = np.unique(coords,axis=0, return_inverse=True)[1]
    if group_synapses:
        unique_synapse_sites = synapse_sites.drop_duplicates( ['x', 'y', 'z'] ).copy()
        all_connecting_bodyIds = synapse_sites['connecting_bodyId'].to_numpy()
        all_connecting_types = synapse_sites['connecting_type'].to_numpy()
        connecting_bodyIds = []; connecting_types = []; mean_coords = []

        for synapse_id in np.unique(synapse_ids):
            synapse_bool = synapse_id == synapse_ids
            connecting_bodyIds.append( all_connecting_bodyIds[ synapse_bool ] )
            connecting_types.append( all_connecting_types[ synapse_bool ] )
            mean_coords.append( np.mean(connecting_coords[synapse_bool],axis=0) )
        unique_synapse_sites[ 'connecting_bodyId' ] = connecting_bodyIds
        unique_synapse_sites[ 'connecting_type' ] = connecting_types
        for i_dim, dim in enumerate(['x','y','z']):
            unique_synapse_sites[ f'connecting_{dim}' ] = np.array(mean_coords)[:,i_dim]
            synapse_sites = unique_synapse_sites.copy()
    return synapse_sites[keep_cols]

def calculate_flipped_and_box(row):
    coord = np.array([row['x'], row['y'], row['z']])
    coord_flipped = np.flip(coord)
    init_box_zyx = np.array([coord_flipped - 50, coord_flipped + 50])
    return pd.Series({'coord': coord, 'init_box_zyx': init_box_zyx})
    
def generate_presynapse_df(neuron_quality_np, out_file='presynapse_data.pkl'):
    presynapse_sub_dfs = []
    for i in range(len(neuron_quality_np)):
        if i % 50 == 0:
            print(i)
        bodyId, neuron_type = neuron_quality_np[i,[0,1]]
        df = utils.get_synapse_df(bodyId, neuron_type, 'pre', group_synapses = True)
        presynapse_sub_dfs.append(df)

    presynapse_df = pd.concat(presynapse_sub_dfs, axis=0)
    
    presynapse_df = presynapse_df[presynapse_df.confidence > 0.95]
    coords_df = presynapse_df.apply(calculate_flipped_and_box, axis=1)
    result_df = pd.concat([presynapse_df, coords_df],axis=1)
    result_df.to_pickle(out_file)
    return result_df
    
