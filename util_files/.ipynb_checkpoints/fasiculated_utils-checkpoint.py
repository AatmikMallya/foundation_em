# % matplotlib inline
from neuprint import Client, fetch_roi_hierarchy, skeleton
from neuprint import fetch_synapses, NeuronCriteria as NC, SynapseCriteria as SC
from neuprint.queries import fetch_mitochondria
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from os import listdir
import importlib
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from os.path import isfile
import statsmodels.api as sm
from scipy.spatial.distance import pdist, squareform, cdist
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from scipy import stats
from matplotlib.patches import Ellipse
import pickle
from sklearn import svm
import networkx as nx

import warnings
warnings.filterwarnings("ignore") # ignore all warnings

token_id = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImdhcnJldHQuc2FnZXJAeWFsZS5lZHUiLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hLS9BT2gxNEdpTGNqZXlHYWNnS3NPcTgzdDNfczBoTU5sQUtlTkljRzdxMkU5Rz1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTgwMTAxNzUwNn0.dzq7Iy01JwSWbKq-Qvi8ov7Hwr0-ozpYeSnOsUD-Mx0"
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
home_dir = '/home/gsager56/hemibrain/clean_mito_code'
c = Client('neuprint.janelia.org', dataset='hemibrain:v1.2.1', token=token_id)
neuron_quality = pd.read_csv(home_dir + '/saved_data/neuron_quality.csv')
neuron_quality_np = neuron_quality.to_numpy()
server = 'http://hemibrain-dvid.janelia.org'

# import utils file
spec = importlib.util.spec_from_file_location('utils', home_dir+'/util_files/utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

# import config file
spec = importlib.util.spec_from_file_location('config', home_dir+'/util_files/config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

analyze_neurons = config.analyze_neurons
node_class_dict = config.node_class_dict

def get_all_paths(G):
    '''
    Loop through all the connected components of G and return the linear paths
    
    '''
    
    all_paths = []
    for cc in nx.connected_components(G):
        degrees = [G.degree[c] for c in cc]
        leaf_nodes = np.array(list(cc))[ np.array(degrees) == 1 ].astype(int)
        paths = []
        for i_leaf in range(len(leaf_nodes)-1):
            for j_leaf in range(i_leaf+1, len(leaf_nodes)):
                shortest_path = nx.shortest_path(G, source=leaf_nodes[i_leaf], target=leaf_nodes[j_leaf])
                if len(shortest_path) > 0:
                    paths.append( shortest_path )
         # longest to shortest_paths
        if len(paths) > 0:
            all_paths.append( paths[ np.argmax([len(path) for path in paths]) ] )
            #all_paths.append( [paths[idx] for idx in np.flip( np.argsort([len(path) for path in paths]) )] )
    return all_paths

def get_matching_paths(i_path, j_paths, i_nodes, j_nodes, dist_thresh, node_dist_matrix):
    max_sum = 0
    best_i_path, best_j_path = None, None
    all_valid_nodes = [ j_nodes[ node_dist_matrix[np.where(i_nodes==i_node)[0][0],:] < dist_thresh ] for i_node in i_path ]
    for j_path in j_paths:
        if len(j_path) > max_sum:
            has_j_path_neigh = [np.any(np.isin(valid_nodes, j_path)) for valid_nodes in all_valid_nodes]
            if np.sum(has_j_path_neigh) > max_sum:
                # save this as the best paths
                best_i_path = np.array(i_path)[np.array(has_j_path_neigh)]

                all_j_valid_nodes = [ i_nodes[ node_dist_matrix[:,np.where(j_nodes==j_node)[0][0]] < dist_thresh ] for j_node in j_path ]
                has_i_path_neigh = [np.any(np.isin(valid_nodes, i_path)) for valid_nodes in all_j_valid_nodes]
                best_j_path = np.array(j_path)[np.array(has_i_path_neigh)]

                max_sum = np.sum(has_j_path_neigh)
                if max_sum >= len(i_path):
                    return best_i_path, best_j_path
    return best_i_path, best_j_path

def get_dists(best_i_path_coords, best_j_path_coords, dx = 0, return_ix = False):
    i_xs = np.cumsum(np.append(0, np.sqrt( np.sum(np.diff(best_i_path_coords,axis=0)**2,axis=1) )))
    j_xs = np.cumsum(np.append(0, np.sqrt( np.sum(np.diff(best_j_path_coords,axis=0)**2,axis=1) ))) + dx
    
    total_dists = []
    for i_node_idx in range(len(i_xs)):
        j_node_idx = np.argmin( (i_xs[i_node_idx] - j_xs)**2 )
        total_dists.append( np.sum( (best_i_path_coords[i_node_idx] - best_j_path_coords[j_node_idx])**2 ) )

    for j_node_idx in range(len(j_xs)):
        i_node_idx = np.argmin( (j_xs[j_node_idx] - i_xs)**2 )
        total_dists.append( np.sum( (best_i_path_coords[i_node_idx] - best_j_path_coords[j_node_idx])**2 ) )
    if return_ix: return total_dists, i_xs, j_xs
    return total_dists

def ensure_axon_left(path, path_coords, mitos_bool, axon_coord):
    '''
    Ensure going left to right in path is axon to dnedrite
    '''
    
    if np.sum((path_coords[0] - axon_coord)**2) > np.sum((path_coords[-1] - axon_coord)**2):
        # the last node is closer to the axon, so flip the path
        path = np.flip(path)
        path_coords = np.flip(path_coords)
        mitos_bool = np.flip(mitos_bool)
    return path, path_coords, mitos_bool

def get_cross_dists(mito_xs, xs, x, side):
    assert side == 'dendrite' or side == 'axon'
    
    this_dists = mito_xs - x if side == 'dendrite' else x - mito_xs
    max_dist = xs[-1]-x if side == 'dendrite' else x - xs[0]
    this_dists = this_dists[this_dists >= 0]
    return this_dists, max_dist

def get_hist(this_dists, max_dist, bins):
    this_hist = np.histogram(this_dists, bins=bins)[0]
    if bins[-1] > max_dist:
        this_hist[ (np.where( bins > max_dist )[0][0] - 1): ] = -1
    return this_hist


def get_best_paths(section, i_neuron, j_neuron, den_dists_maxDist, axon_dists_maxDist, method, dist_thresh = 2):
    i_bodyId, i_neuron_type = neuron_quality_np[i_neuron,[0,1]]
    i_skel_file = home_dir + f'/saved_clean_skeletons/s_pandas_{i_bodyId}_{i_neuron_type}_200nm.csv'
    i_mito_file =  home_dir + f'/saved_mito_df/{i_neuron_type}_{i_bodyId}_mito_df.csv'
    
    if not isfile(i_mito_file):
        return den_dists_maxDist, axon_dists_maxDist
    
    j_bodyId, j_neuron_type = neuron_quality_np[j_neuron,[0,1]]
    j_skel_file = home_dir + f'/saved_clean_skeletons/s_pandas_{j_bodyId}_{j_neuron_type}_200nm.csv'
    j_mito_file =  home_dir + f'/saved_mito_df/{j_neuron_type}_{j_bodyId}_mito_df.csv'
    
    if not isfile(j_mito_file):
        return den_dists_maxDist, axon_dists_maxDist
    
    i_s_pandas = pd.read_csv( i_skel_file )
    i_nodes = i_s_pandas['rowId'].to_numpy()
    i_nx_graph = skeleton.skeleton_df_to_nx(i_s_pandas, with_attributes=True, directed=False)
    i_mito_df = pd.read_csv( i_mito_file )
    i_coords = i_s_pandas.to_numpy()[:,[1,2,3]] * 8/1000 # um

    i_mito_nodes = i_nodes[ utils.find_closest_idxs(i_s_pandas.to_numpy(), i_mito_df) ]

    j_s_pandas = pd.read_csv( j_skel_file )
    j_nodes = j_s_pandas['rowId'].to_numpy()
    j_coords = j_s_pandas.to_numpy()[:,[1,2,3]] * 8/1000 # um
    j_nx_graph = skeleton.skeleton_df_to_nx(j_s_pandas, with_attributes=True, directed=False)
    j_mito_df = pd.read_csv( j_mito_file )
    j_mito_nodes = j_nodes[ utils.find_closest_idxs(j_s_pandas.to_numpy(), j_mito_df) ]
    node_dist_matrix = cdist(i_coords, j_coords)
    
    i_nodes_delete = i_nodes[i_s_pandas['node_classes'].to_numpy() != node_class_dict[section]]
    if method == 'measured':
        #print(len(np.min(node_dist_matrix,axis=1) > (dist_thresh*i_s_pandas.to_numpy()[:,4] * 8/1000*2)), len(i_nodes))
        i_nodes_delete = np.unique(np.append(i_nodes_delete, i_nodes[np.min(node_dist_matrix,axis=1) > (dist_thresh*i_s_pandas.to_numpy()[:,4] * 8/1000*2)]))
        
    for i_node in i_nodes_delete:
        i_nx_graph.remove_node(i_node)
    j_nodes_delete = j_nodes[j_s_pandas['node_classes'].to_numpy() != node_class_dict[section]]
    if method == 'measured':
        j_nodes_delete = np.unique(np.append(j_nodes_delete, j_nodes[np.min(node_dist_matrix,axis=0) > (dist_thresh*j_s_pandas.to_numpy()[:,4] * 8/1000*2)]))
    for j_node in j_nodes_delete:
        j_nx_graph.remove_node(j_node)

    i_paths = get_all_paths(i_nx_graph)
    j_paths = get_all_paths(j_nx_graph)
    
    for i_path in i_paths:
        if method == 'shuffled':
            best_i_path_coords = np.array([ i_coords[ np.where(node == i_nodes)[0][0] ] for node in i_path ])
            j_path = np.array(j_paths[0])
            best_j_path_coords = np.array([ j_coords[ np.where(node == j_nodes)[0][0] ] for node in j_path ])
            
            i_xs = np.cumsum(np.append(0, np.sqrt( np.sum(np.diff(best_i_path_coords,axis=0)**2,axis=1) )))
            j_xs = np.cumsum(np.append(0, np.sqrt( np.sum(np.diff(best_j_path_coords,axis=0)**2,axis=1) )))
            min_length = 10 # um
            max_length = 30 # um
            
            if i_xs[-1] > min_length and j_xs[-1] > min_length:
                # continue with analysis
                i_path = np.array(i_path)[ np.random.choice(np.where( (i_xs - i_xs[-1]) < 0 )[0]): ]
                j_path = j_path[ np.random.choice(np.where( (j_xs - j_xs[-1]) < 0 )[0]): ]
            
            best_i_path_coords = np.array([ i_coords[ np.where(node == i_nodes)[0][0] ] for node in i_path ])
            best_j_path_coords = np.array([ j_coords[ np.where(node == j_nodes)[0][0] ] for node in j_path ])
            i_xs = np.cumsum(np.append(0, np.sqrt( np.sum(np.diff(best_i_path_coords,axis=0)**2,axis=1) )))
            j_xs = np.cumsum(np.append(0, np.sqrt( np.sum(np.diff(best_j_path_coords,axis=0)**2,axis=1) )))
            best_i_path = i_path[ i_xs < max_length ]
            best_j_path = j_path[ j_xs < max_length ]
                
        else:
            best_i_path, best_j_path = get_matching_paths(i_path, j_paths, i_nodes, j_nodes, dist_thresh, node_dist_matrix)
        if best_i_path is not None:
            i_mitos_bool = np.isin(best_i_path, i_mito_nodes)
            j_mitos_bool = np.isin(best_j_path, j_mito_nodes)

            if np.any(i_mitos_bool) and np.any(j_mitos_bool) and (np.sum(i_mitos_bool) + np.sum(j_mitos_bool) >= 5):

                best_i_path_coords = np.array([ i_coords[ np.where(node == i_nodes)[0][0] ] for node in best_i_path ])
                best_j_path_coords = np.array([ j_coords[ np.where(node == j_nodes)[0][0] ] for node in best_j_path ])

                axon_coord = np.mean( i_coords[ i_s_pandas['node_classes'].to_numpy() == node_class_dict['axon'] ], axis=0)
                best_i_path, best_i_path_coords, i_mitos_bool = ensure_axon_left(best_i_path, best_i_path_coords, i_mitos_bool, axon_coord)

                axon_coord = np.mean( j_coords[ j_s_pandas['node_classes'].to_numpy() == node_class_dict['axon'] ], axis=0)
                best_j_path, best_j_path_coords, j_mitos_bool = ensure_axon_left(best_j_path, best_j_path_coords, j_mitos_bool, axon_coord)
                
                dists, i_xs, j_xs = get_dists(best_i_path_coords, best_j_path_coords, return_ix = True)
                
                if method == 'measured':
                    fun = lambda x: np.sum(get_dists(best_i_path_coords, best_j_path_coords, dx=x))
                    dxs = np.linspace(-i_xs[-1]/2,i_xs[-1]/2,100)
                    dys = [fun(dx) for dx in dxs]
                    dxs = np.linspace(dxs[0]-dxs[1],dxs[0]-dxs[1],100) + dxs[np.argmin(dys)]
                    dys = [fun(dx) for dx in dxs]
                    dx = dxs[np.argmin(dys)]
                    
                for ix in i_xs[i_mitos_bool]:
                    den_dists_maxDist.append( get_cross_dists(j_xs[j_mitos_bool], j_xs, ix, 'dendrite') )
                    axon_dists_maxDist.append(get_cross_dists(j_xs[j_mitos_bool], j_xs, ix, 'axon') )
                for jx in j_xs[j_mitos_bool]:
                    den_dists_maxDist.append( get_cross_dists(i_xs[i_mitos_bool], i_xs, jx, 'dendrite') )
                    axon_dists_maxDist.append(get_cross_dists(i_xs[i_mitos_bool], i_xs, jx, 'axon') )
    return den_dists_maxDist, axon_dists_maxDist