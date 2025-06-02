import numpy as np
import pandas as pd

def clean_false_branches(neuron):
    """
    Clean false branches in neuron skeletonization by removing branches shorter than
    the neurite diameter. This function specifically looks for cases where multiple
    nodes connect to the same target, which creates branch points when constructing
    a directed graph.
    
    Args:
        neuron: DataFrame with columns [x, y, z, rowId, radius, link]
        
    Returns:
        tuple: (cleaned_df, pruned_nodes_df)
            - cleaned_df: DataFrame with false branches removed
            - pruned_nodes_df: DataFrame containing the nodes that were pruned
    """
    # Create a copy to work with
    df = neuron.copy()
    
    # First, identify branch points by finding nodes that have the same target (link value)
    # This indicates that multiple nodes are connecting to the same target node
    link_series = df['link'].copy()
    link_series = link_series[link_series != -1]  # Remove terminal nodes
    duplicate_mask = link_series.duplicated(keep=False)
    duplicated_links = link_series[duplicate_mask]
    
    # Get the unique link values that have multiple source nodes
    branch_targets = duplicated_links.unique()
    print(f"Found {len(branch_targets)} potential branch points")
    
    # Edges to remove (false branches)
    edges_to_remove = []
    # Detailed info about pruned branches
    pruned_info = []
    
    # Process each branch point
    for branch_target in branch_targets:
        # Get all source nodes connecting to this target
        source_nodes = df[df['link'] == branch_target]['rowId'].tolist()
        target_radius = df[df['rowId'] == branch_target]['radius'].values[0]
        target_diameter = 2 * target_radius
        
        # If there's only one source, this isn't actually a branch point
        if len(source_nodes) <= 1:
            continue
            
        # Get coordinates for each source node
        source_coords = []
        for src in source_nodes:
            row = df[df['rowId'] == src]
            source_coords.append((
                row['rowId'].values[0],
                row['x'].values[0], 
                row['y'].values[0], 
                row['z'].values[0]
            ))
        
        # For each pair of source nodes, compute the distance
        false_branches = []
        for i, (src1_id, x1, y1, z1) in enumerate(source_coords):
            # Calculate distance to all other sources
            for j, (src2_id, x2, y2, z2) in enumerate(source_coords[i+1:], i+1):
                dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                
                # If distance is less than diameter, one of these is likely a false branch
                if dist < target_diameter:
                    # Store the source node as a false branch
                    false_branches.append(src2_id)
                    
                    # Add detailed info about this pruned branch
                    pruned_info.append({
                        'src_id': src2_id,
                        'target_id': branch_target,
                        'distance': dist,
                        'target_diameter': target_diameter,
                        'reason': 'Distance between source nodes less than branch point diameter'
                    })
        
        # Add edges to remove
        for src in false_branches:
            edges_to_remove.append((src, branch_target))
    
    print(f"Identified {len(edges_to_remove)} false branches")
    
    # Update the dataframe by removing the links for false branches
    result_df = df.copy()
    
    # Track pruned nodes
    pruned_rows = []
    
    # For each false branch, set its link to -1 (disconnected)
    for src_id, target_id in edges_to_remove:
        idx = result_df[result_df['rowId'] == src_id].index
        if len(idx) > 0:
            # Store the original row before modifying
            pruned_rows.append(result_df.loc[idx[0]].copy())
            # Set link to -1
            result_df.at[idx[0], 'link'] = -1
    
    # Create a dataframe of pruned nodes
    pruned_nodes_df = pd.DataFrame(pruned_rows) if pruned_rows else pd.DataFrame()
    
    # Add the pruning info (reason, distance, etc.)
    if not pruned_nodes_df.empty and pruned_info:
        pruned_info_df = pd.DataFrame(pruned_info)
        # Create a mapping from src_id to pruning info
        pruning_info_map = {row['src_id']: row for row in pruned_info}
        
        # Add columns for pruning details
        pruned_nodes_df['pruning_reason'] = pruned_nodes_df['rowId'].map(
            lambda src_id: pruning_info_map.get(src_id, {}).get('reason', 'Unknown')
        )
        pruned_nodes_df['distance_to_other_source'] = pruned_nodes_df['rowId'].map(
            lambda src_id: pruning_info_map.get(src_id, {}).get('distance', np.nan)
        )
        pruned_nodes_df['target_diameter'] = pruned_nodes_df['rowId'].map(
            lambda src_id: pruning_info_map.get(src_id, {}).get('target_diameter', np.nan)
        )
    
    return result_df, pruned_nodes_df 