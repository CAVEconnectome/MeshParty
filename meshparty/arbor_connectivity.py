import numpy as np
from meshparty import utils, skeleton
from meshparty.trimesh_io import MaskedMesh
from scipy import spatial, sparse
from pykdtree.kdtree import KDTree as pyKDTree
from copy import copy

def annotation_location_indices(mesh, anno_df, pos_column, sk_map=None, max_dist=np.inf,
                                voxel_resolution=np.array([4,4,40]), mesh_index_col_name='mind',
                                skeleton_index_col_name='skind'):
    '''
    For a synapse dataframe associated with a given neuron, find the mesh indices associated with each synapse.

    :param mesh: trimesh Mesh or MaskedMesh
    :param synapse_df: DataFrame with at least one position column
    :param pos_column: string, column of dataframe to use for annotation positions
    :param sk_map: Optional, Numpy array with skeleton vertex index for every mesh vertex index.
    :param max_dist: Optional, Maximum distance to the mesh allowed for assignment, else return -1.
    :param voxel_resolution: Optional, default is [4,4,40] nm/voxel.
    :param mesh_index_col_name: Optional, string of new mesh index column name. Default 'mind'
    :param skeleton_index_col_name: Optional, string of new skeleton index column name. Default 'skind'
    :returns: Copy of anno_df with additional column(s) for the (unmasked) mesh index and, if created, skeleton index.
    '''
    anno_positions = np.vstack(anno_df[pos_column].values) * voxel_resolution
    ds, mesh_inds = mesh.pykdtree.query(anno_positions)
    mesh_inds[ds>max_dist] = -1
    
    anno_df = anno_df.copy()
    if type(mesh) is MaskedMesh:
        anno_df[mesh_index_col_name] = mesh.map_indices_to_unmasked(mesh_inds)
    else:
        anno_df[mesh_index_col_name] = mesh_inds

    if sk_map is not None:
        sk_map=sk_map.astype(int)
        skinds = sk_map[mesh_inds]
        skinds[ds>max_dist] = -1
        anno_df[skeleton_index_col_name] = skinds
    return anno_df

def annotation_skeleton_segments(sk, anno_df, pos_column, mesh=None, anno_skind_col=None, max_dist=np.inf,
                                 voxel_resolution=np.array([4,4,40]), skeleton_index_col_name='skind',
                                 mesh_index_col_name='mind', skeleton_segment_col_name='seg_ind'):
    '''
    Attach skeleton segment index to an annotaiton dataframe
    '''
    if mesh is None and anno_skind_col is None:
        raise ValueError('Must have either a mesh or existing skeleton indices')        
    sk_map = sk.mesh_to_skel_map[mesh.node_mask]
    if anno_skind_col is None:
        anno_result_df = annotation_location_indices(sk_map, mesh, anno_df,
                                                     max_dist=max_dist, voxel_resolution=voxel_resolution,
                                                     pos_column=pos_column, mesh_index_col_name=mesh_index_col_name,
                                                     skeleton_index_col_name=skeleton_index_col_name)
        anno_result_df[skeleton_segment_col_name] = sk.segment_map[anno_result_df[skeleton_index_col_name]]
    else:
        anno_result_df = anno_df.copy()
        anno_result_df[skeleton_segment_col_name] = sk.segment_map[anno_result_df[skeleton_index_col_name]]
    return syn_result_df


def skind_to_anno_map(sk, mesh, anno_df, pos_column, mesh=None, anno_skind_col=None, max_dist=np.inf,
                      voxel_resolution=np.array([4,4,40]), skeleton_index_col_name='skind',
                      mesh_index_col_name='mind', skeleton_segment_col_name='seg_ind'):
    '''
    Make a dict with key skeleton index and values a list of annotation ids at that index.
    '''
    anno_dict = {}
    if len(anno_df) == 0:
        return anno_dict

    anno_sk_df = annotation_skeleton_segments(sk, mesh, anno_df,
                                              max_dist=max_dist, voxel_resolution=voxel_resolution,
                                              pos_column=pos_column, mesh_index_col_name=mesh_index_col_name,
                                              skeleton_index_col_name=skeleton_index_col_name)
    for k, v in anno_sk_df[[skeleton_index_col_name, 'id']].groupby(skeleton_index_col_name).agg(lambda x: [int(y) for y in x]).to_dict()['id'].items():
        anno_dict[k] = v
    return anno_dict


def skeleton_segment_df(sk, anno_df, skeleton_segment_col_name='seg_ind', keep_columns=[]):
    '''
    Make a dataframe where every row is a segment of the skeleton.
    '''
    if keep_columns is None:
        keep_columns_default = ['id', 'pt_root_id', 'pre_pt_root_id', 'post_pt_root_id', 'ctr_pt_position', 'size', 'mind', 'skind', skeleton_segment_col_name]
        keep_columns = anno_df.columns[np.isin(anno_df.columns, keep_columns_default)]

    anno_df = anno_df[keep_columns].groupby(skeleton_segment_col_name).agg(list).reset_index()
    return seg_syn_df
