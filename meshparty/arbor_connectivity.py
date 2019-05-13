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


def synapse_betweenness(sk, pre_inds, post_inds):
    '''
    Compute synapse betweeness (number of paths from an input to an output) for all nodes of a skeleton.
    :param sk: Skeleton
    :param pre_inds: List of skeleton indices with an input synapse (Duplicate indices with more than one)
    :param post_inds: List of skeleton indices with an output synapse (Duplicate indices with more than one)
    '''
    def _precompute_synapse_inds(syn_inds):
        Nsyn = len(syn_inds)
        n_syn = np.zeros(len(sk.vertices))
        for ind in syn_inds:
            n_syn[ind] += 1
        return Nsyn, n_syn

    Npre, n_pre = _precompute_synapse_inds(pre_inds)
    Npost, n_post = _precompute_synapse_inds(post_inds)
    
    syn_btwn = np.zeros(len(sk.vertices))
    cov_paths_rev = sk.covering_paths[::-1]
    for path in cov_paths_rev:
        downstream_pre = 0
        downstream_post = 0
        for ind in path:
            downstream_pre += n_pre[ind]
            downstream_post += n_post[ind]
            syn_btwn[ind] = downstream_pre * (Npost - downstream_post) + \
                               (Npre - downstream_pre) * downstream_post
        # Deposit each branch's synapses at the branch point.
        bp_ind = sk.parent_node(path[-1])
        if bp_ind is not None:
            n_pre[bp_ind]  += downstream_pre
            n_post[bp_ind] += downstream_post
    return syn_btwn


def split_axon_by_synapse_betweenness(sk, pre_inds, post_inds, return_quality=True, extend_to_segment=True):
    '''
    Find the is_axon boolean label for all vertices in the skeleton. Assumes skeleton root is not on the axon side.
    :param sk: Skeleton
    :param pre_inds: List of skeleton indices with an input synapse (Duplicate indices with more than one)
    :param post_inds: List of skeleton indices with an output synapse (Duplicate indices with more than one)
    :param return_quality: Compute split quality at the split point. Always computer for true split point, not shifted one.
    :param extend_to_segment: Shift split point to the closest-to-root location on the same segment as the split node.
    :returns: boolean array, True for axon vertices.
    :returns: float, optional split quality index.
    '''

    axon_split = find_axon_split_vertex_by_synapse_betweenness(sk, pre_inds, post_inds, return_quality=return_quality, extend_to_segment=True)
    if return_quality:
        axon_split_ind, split_quality = axon_split
    else:
        axon_split_ind = axon_split
    is_axon = np.full(len(sk.vertices), False)
    is_axon[ sk.downstream_nodes(axon_split_ind) ] = True
    
    if return_quality:
        return is_axon, split_quality
    else:
        return is_axon


def find_axon_split_vertex_by_synapse_betweenness(sk, pre_inds, post_inds, return_quality=True, extend_to_segment=True):
    '''
    Find the skeleton vertex at which to split the axon from the dendrite. Assumes skeleton root is on dendritic side.
    :param sk: Skeleton
    :param pre_inds: List of skeleton indices with an input synapse (Duplicate indices with more than one)
    :param post_inds: List of skeleton indices with an output synapse (Duplicate indices with more than one)
    :param return_quality: Compute split quality at the split point. Always computer for true split point, not shifted one.
    :param extend_to_segment: Shift split point to the closest-to-root location on the same segment as the split node.
    :returns: int, skeleton index
    :returns: float, optional split quality index.
    '''
    syn_btwn = synapse_betweenness(sk, pre_inds, post_inds)
    high_vinds = np.flatnonzero(syn_btwn==max(syn_btwn))
    close_vind = high_vinds[np.argmin(sk.distance_to_root[high_vinds])]
    
    if return_quality:
        axon_qual_label = np.full(len(sk.vertices), False)
        axon_qual_label[ sk.downstream_nodes(close_vind) ] = True
        split_quality = axon_split_quality(axon_qual_label, pre_inds, post_inds)

    if extend_to_segment:
        relseg = sk.segment_map[close_vind]
        axon_split_ind = sk.segments[relseg][-1]
    else:
        axon_split_ind = close_vind
    
    if return_quality:
        return axon_split_ind, split_quality
    else:
        return axon_split_ind


def axon_split_quality(is_axon, pre_inds, post_inds):
    '''
    Returns a quality index (0-1, higher is a cleaner split) for split quality,
    defined as best separating input and output sites.
    '''
    axon_pre = sum(is_axon[pre_inds])
    axon_post = sum(is_axon[post_inds])
    dend_pre = sum(~is_axon[pre_inds])
    dend_post = sum(~is_axon[post_inds])

    counts = np.array([[axon_pre, axon_post],[dend_pre, dend_post]])
    observed_ent = _distribution_split_entropy( counts )
    
    unsplit_ent = _distribution_split_entropy([[len(pre_inds), len(post_inds)]] )

    return 1-observed_ent/unsplit_ent


def _distribution_split_entropy( counts ):
    if np.sum(counts)==0:
        return 0
    ps = np.divide(counts, np.sum(counts, axis=1)[:,np.newaxis], where=np.sum(counts, axis=1)[:,np.newaxis]>0)
    Hpart = np.sum(np.multiply(ps, np.log2(ps, where=ps>0)), axis=1)
    Hws = np.sum(counts, axis=1) / np.sum(counts)
    Htot = -np.sum(Hpart * Hws)
    return Htot
