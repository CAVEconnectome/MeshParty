import numpy as np
from meshparty import utils, skeleton
from copy import copy


def annotation_location_indices(
    mesh,
    anno_df,
    pos_column,
    sk_map=None,
    max_dist=np.inf,
    voxel_resolution=np.array([4, 4, 40]),
):
    """
    For a synapse dataframe associated with a given neuron, find the mesh indices associated with each synapse.

    :param mesh: trimesh Mesh
    :param synapse_df: DataFrame with at least one position column
    :param pos_column: string, column of dataframe to use for annotation positions
    :param sk_map: Optional, Numpy array with skeleton vertex index for every mesh vertex index.
    :param max_dist: Optional, Maximum distance to the mesh allowed for assignment, else return -1.
    :param voxel_resolution: Optional, default is [4,4,40] nm/voxel.
    :returns: Mesh indices and, if desired, skeleton indices.
    """
    if len(anno_df) == 0:
        if sk_map is None:
            return np.array([])
        else:
            return np.array([]), np.array([])

    anno_positions = np.vstack(anno_df[pos_column].values) * voxel_resolution
    ds, mesh_inds = mesh.kdtree.query(anno_positions)
    mesh_inds[ds > max_dist] = -1

    if sk_map is None:
        return mesh_inds
    else:
        sk_map = sk_map.astype(int)
        found_inds = mesh_inds >= 0
        skinds = np.zeros(mesh_inds.shape)
        skinds[found_inds] = sk_map[mesh_inds[found_inds]]
        skinds[~found_inds] = -1
        return mesh_inds, skinds


def annotation_skeleton_segments(
    sk,
    anno_df,
    pos_column,
    mesh=None,
    max_dist=np.inf,
    voxel_resolution=np.array([4, 4, 40]),
    skeleton_index_col_name=None,
):
    """
    Attach skeleton segment index to an annotation dataframe
    :param sk: Skeleton
    :param anno_df: Annotation dataframe
    :param pos_column: String. Column name in dataframe with position values
    :param mesh: optional, mesh object. Needed if skeleton_index_col_name is not specified.
    :param max_dist: optional, float. Max distance to mesh for attaching annotations. Default is inf.
    :param voxel_resolution: optional, length 3 array. Default is [4,4,40] nm/pixel.
    :param skeleton_index_col_name: optional, string. Column name of skeleton vertex in dataframe.
    """
    if mesh is None and skeleton_index_col_name is None:
        raise ValueError("Must have either a mesh or existing skeleton indices")

    if skeleton_index_col_name is None:
        sk_map = sk.mesh_to_skel_map
        minds, skinds = annotation_location_indices(
            mesh,
            anno_df,
            pos_column,
            sk_map=sk_map,
            max_dist=max_dist,
            voxel_resolution=voxel_resolution,
        )
        anno_segments = sk.segment_map[skinds]
        return anno_segments, minds, skinds
    else:
        anno_segments = sk.segment_map[anno_df[skeleton_index_col_name]]
        return anno_segments


def skind_to_anno_map(
    sk,
    anno_df,
    pos_column=None,
    mesh=None,
    max_dist=np.inf,
    voxel_resolution=np.array([4, 4, 40]),
    skeleton_index_col_name=None,
):
    """
    Make a dict with key skeleton index and values a list of annotation ids at that index.
    """
    anno_dict = {}
    if len(anno_df) == 0:
        return anno_dict

    if skeleton_index_col_name is None:
        minds, skinds = annotation_location_indices(
            mesh,
            anno_df,
            pos_column,
            sk_map=sk.mesh_to_skel_map,
            max_dist=max_dist,
            voxel_resolution=voxel_resolution,
        )
        anno_df = anno_df.copy()
        skeleton_index_col_name = "XXX_temp_skeleton_index_internal"
        anno_df[skeleton_index_col_name] = skinds

    for k, v in (
        anno_df[[skeleton_index_col_name, "id"]]
        .groupby(skeleton_index_col_name)
        .agg(lambda x: [int(y) for y in x])
        .to_dict()["id"]
        .items()
    ):
        anno_dict[int(k)] = v
    return anno_dict


def synapse_betweenness(sk, pre_inds, post_inds):
    """
    Compute synapse betweeness (number of paths from an input to an output) for all nodes of a skeleton.
    :param sk: Skeleton
    :param pre_inds: List of skeleton indices with an input synapse (Duplicate indices with more than one)
    :param post_inds: List of skeleton indices with an output synapse (Duplicate indices with more than one)

    :returns: Array of synapse betweenness for every vertex in the skeleton.
    """

    def _precompute_synapse_inds(syn_inds):
        Nsyn = len(syn_inds)
        n_syn = np.zeros(len(sk.vertices))
        for ind in syn_inds:
            n_syn[ind] += 1
        return Nsyn, n_syn

    Npre, n_pre = _precompute_synapse_inds(pre_inds)
    Npost, n_post = _precompute_synapse_inds(post_inds)

    syn_btwn = np.zeros(len(sk.vertices))
    cov_paths_rev = sk.cover_paths[::-1]
    for path in cov_paths_rev:
        downstream_pre = 0
        downstream_post = 0
        for ind in path:
            downstream_pre += n_pre[ind]
            downstream_post += n_post[ind]
            syn_btwn[ind] = (
                downstream_pre * (Npost - downstream_post)
                + (Npre - downstream_pre) * downstream_post
            )
        # Deposit each branch's synapses at the branch point.
        bp_ind = sk.parent_node(path[-1])
        if bp_ind is not None:
            n_pre[bp_ind] += downstream_pre
            n_post[bp_ind] += downstream_post
    return syn_btwn


def split_axon_by_synapse_betweenness(
    sk, pre_inds, post_inds, return_quality=True, extend_to_segment=True
):
    """
    Find the is_axon boolean label for all vertices in the skeleton. Assumes skeleton root is not on the axon side.
    :param sk: Skeleton
    :param pre_inds: List of skeleton indices with an input synapse (Duplicate indices with more than one)
    :param post_inds: List of skeleton indices with an output synapse (Duplicate indices with more than one)
    :param return_quality: Compute split quality at the split point. Always computer for true split point, not shifted one.
    :param extend_to_segment: Shift split point to the closest-to-root location on the same segment as the split node.
    :returns: boolean array, True for axon vertices.
    :returns: float, optional split quality index.
    """
    pre_inds = _check_ind_list(pre_inds)
    post_inds = _check_ind_list(post_inds)

    axon_split = find_axon_split_vertex_by_synapse_betweenness(
        sk, pre_inds, post_inds, return_quality=return_quality, extend_to_segment=True
    )
    if return_quality:
        axon_split_ind, split_quality = axon_split
    else:
        axon_split_ind = axon_split
    is_axon = np.full(len(sk.vertices), False)
    is_axon[sk.downstream_nodes(axon_split_ind)] = True

    if return_quality:
        return is_axon, split_quality
    else:
        return is_axon


def find_axon_split_vertex_by_synapse_betweenness(
    sk, pre_inds, post_inds, return_quality=True, extend_to_segment=True
):
    """
    Find the skeleton vertex at which to split the axon from the dendrite. Assumes skeleton root is on dendritic side.
    :param sk: Skeleton
    :param pre_inds: List of skeleton indices with an input synapse (Duplicate indices with more than one)
    :param post_inds: List of skeleton indices with an output synapse (Duplicate indices with more than one)
    :param return_quality: Compute split quality at the split point. Always computer for true split point, not shifted one.
    :param extend_to_segment: Shift split point to the closest-to-root location on the same segment as the split node.
    :returns: int, skeleton index
    :returns: float, optional split quality index.
    """
    pre_inds = _check_ind_list(pre_inds)
    post_inds = _check_ind_list(post_inds)

    syn_btwn = synapse_betweenness(sk, pre_inds, post_inds)
    high_vinds = np.flatnonzero(syn_btwn == max(syn_btwn))
    close_vind = high_vinds[np.argmin(sk.distance_to_root[high_vinds])]

    if return_quality:
        axon_qual_label = np.full(len(sk.vertices), False)
        axon_qual_label[sk.downstream_nodes(close_vind)] = True
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
    """
    Returns a quality index (0-1, higher is a cleaner split) for split quality,
    defined as best separating input and output sites.
    """
    pre_inds = _check_ind_list(pre_inds)
    post_inds = _check_ind_list(post_inds)

    axon_pre = sum(is_axon[pre_inds])
    axon_post = sum(is_axon[post_inds])
    dend_pre = sum(~is_axon[pre_inds])
    dend_post = sum(~is_axon[post_inds])

    counts = np.array([[axon_pre, axon_post], [dend_pre, dend_post]])
    observed_ent = _distribution_split_entropy(counts)

    unsplit_ent = _distribution_split_entropy([[len(pre_inds), len(post_inds)]])

    return 1 - observed_ent / unsplit_ent


def _distribution_split_entropy(counts):
    if np.sum(counts) == 0:
        return 0
    ps = np.divide(
        counts,
        np.sum(counts, axis=1)[:, np.newaxis],
        where=np.sum(counts, axis=1)[:, np.newaxis] > 0,
    )
    Hpart = np.sum(np.multiply(ps, np.log2(ps, where=ps > 0)), axis=1)
    Hws = np.sum(counts, axis=1) / np.sum(counts)
    Htot = -np.sum(Hpart * Hws)
    return Htot


def _check_ind_list(inds):
    if type(inds) is dict:
        return np.concatenate([[k] * len(v) for k, v in inds.items() if k >= 0])
    else:
        return inds
