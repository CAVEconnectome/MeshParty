import numpy as np
from .utils import window_matrix

#####################
# Split by synapses #
#####################


def split_axon_by_synapses(
    nrn,
    pre_inds,
    post_inds,
    return_quality=True,
    extend_to_segment=True,
):
    """Split a skeleton where the most paths flow between pre and postsynaptic synapse points.

    This method assumes that the axon can be split off by a single skeleton point.

    Parameters
    ----------
    sk : skeleton.Skeleton
        Skeleton with N vertices to split
    pre_inds : array
        Indices of presynaptic sites
    post_inds : array
        Indices of postsynaptic sites
    return_quality : bool, optional
        If True (default), also returns the split quality
    extend_to_segment : bool, optional
        If True (default), places the cut at the proximal base of the segment where the max flow occurs.
        Probably does not make sense to use if the soma is not root.
    n_times : int, optional
        Number of successive times to run the algoritm. This should be the number of distinct axon initial segments a neuron has.

    Returns
    -------
    is_axon : N array
        Boolean array that is True on the axon (presynaptic) side of the neuron.
    split_quality : float
        If return_quality is True, this is a number between 0 and 1 measuring how well the split
        segregates inputs from outputs. 1 is fully segregated, 0 is fully mixed.
    """
    if nrn.skeleton is None:
        raise ValueError("Meshwork must have skeleton")

    pre_inds = nrn._convert_to_meshindex(pre_inds)
    post_inds = nrn._convert_to_meshindex(post_inds)
    pre_inds = pre_inds[pre_inds.to_skel_index_padded >= 0]
    post_inds = post_inds[post_inds.to_skel_index_padded >= 0]

    axon_split = _find_axon_split(
        nrn.skeleton,
        pre_inds.to_skel_index_padded,
        post_inds.to_skel_index_padded,
        return_quality=return_quality,
        extend_to_segment=extend_to_segment,
    )

    if return_quality:
        axon_split_ind, split_quality = axon_split
    else:
        axon_split_ind = axon_split
    downstream_inds = nrn.skeleton.downstream_nodes(axon_split_ind)
    n_pre_ds = np.sum(np.isin(pre_inds.to_skel_index_padded, downstream_inds))
    n_post_ds = np.sum(np.isin(post_inds.to_skel_index_padded, downstream_inds))
    n_pre_us = len(pre_inds) - n_pre_ds
    n_post_us = len(post_inds) - n_post_ds

    # Axon has the higher of the two fractions of pre:
    if (n_pre_ds / (n_post_ds+n_pre_ds+1)) >= (n_pre_us / (n_post_us+n_pre_us+1)):
        is_axon_sk = np.full(len(nrn.skeleton.vertices), False)
        is_axon_sk[downstream_inds] = True
    else:
        is_axon_sk = np.full(len(nrn.skeleton.vertices), True)
        is_axon_sk[nrn.skeleton.downstream_nodes(axon_split_ind)] = False

    is_axon = nrn.SkeletonIndex(np.flatnonzero(is_axon_sk)).to_mesh_index

    if return_quality:
        return is_axon, split_quality
    else:
        return is_axon


def split_axon_by_annotation(
    nrn,
    pre_anno,
    post_anno,
    return_quality=True,
    extend_to_segment=True,
    n_times=1,
):
    all_is_axon = []
    mask = nrn.mesh_mask
    for _ in range(n_times):
        with nrn.mask_context(mask) as nrnm:
            is_axon = split_axon_by_synapses(
                nrnm,
                nrnm.anno[pre_anno].mesh_index,
                nrnm.anno[post_anno].mesh_index,
                return_quality=False,
                extend_to_segment=extend_to_segment,
            )
        all_is_axon.append(is_axon.to_mesh_index_base)
        mask = np.logical_and(mask, np.invert(is_axon.to_mesh_mask_base))

    is_axon_all = nrn.MeshIndex(
        nrn.mesh.filter_unmasked_indices(np.concatenate(all_is_axon))
    )

    if return_quality:
        split_quality = axon_split_quality(
            is_axon_all.to_mesh_mask,
            nrn.anno[pre_anno].mesh_index,
            nrn.anno[post_anno].mesh_index,
        )
        return is_axon_all, split_quality
    else:
        return is_axon_all


def axon_split_quality(is_axon, pre_inds, post_inds):
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


def _precompute_synapse_inds(sk, syn_inds):
    Nsyn = len(syn_inds)
    n_syn = np.zeros(len(sk.vertices))
    for ind in syn_inds:
        n_syn[ind] += 1
    return Nsyn, n_syn


def synapse_betweenness(sk, pre_inds, post_inds, use_entropy=False):
    """ Compute synapse betweenness, the number of paths from all post indices to all pre indices along the graph. Vertices can be included multiple times, indicating multiple paths

    Parameters
    ----------
    sk : Skeleton   
        Skeleton to measure
    pre_inds : list or array
        Collection of skeleton vertex indices, each representing one output synapse (i.e. target of a path).
    post_inds : list or array
        Collection of skeleton certex indices, each representing one input synapse (i.e. source of a path).
    use_entropy : bool, optional
        If True, also returns the entropic segregatation index if one were to cut at a given vertex, by default False

    Returns
    -------
    synapse_betweenness : np.array
        Array with a value for each skeleton vertex, with the number of all paths from source to target vertices passing through that vertex.
    segregation_index : np.array (optional)
        Array with a value for each skeleton vertex, with the segregatio index if the cut were to happen at that vertex. Only returned if `use_entropy=True`.
    """
    Npre, n_pre = _precompute_synapse_inds(sk, pre_inds)
    Npost, n_post = _precompute_synapse_inds(sk, post_inds)

    syn_btwn = np.zeros(len(sk.vertices))
    split_index = np.zeros(len(sk.vertices))
    cov_paths_rev = sk.cover_paths[::-1]
    if use_entropy:
        entropy_normalization = _distribution_split_entropy(
            np.array([[Npre, Npost], [0, 0]])
        )
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
            if use_entropy:
                counts = np.array(
                    [
                        [downstream_pre, downstream_post],
                        [Npost - downstream_post, Npre - downstream_pre],
                    ]
                )
                split_index[ind] = (
                    1 - _distribution_split_entropy(counts) / entropy_normalization
                )
        # Deposit each branch's synapses at the branch point.
        bp_ind = sk.parent_nodes(path[-1])
        if bp_ind is not None:
            n_pre[bp_ind] += downstream_pre
            n_post[bp_ind] += downstream_post
    if use_entropy:
        return syn_btwn, split_index
    else:
        return syn_btwn


def _find_axon_split(
    sk, pre_inds, post_inds, return_quality=True, extend_to_segment=True
):
    syn_btwn = synapse_betweenness(sk, pre_inds, post_inds)
    high_vinds = np.flatnonzero(syn_btwn == max(syn_btwn))
    close_vind = high_vinds[np.argmin(sk.distance_to_root[high_vinds])]

    if return_quality:
        axon_qual_label = np.full(len(sk.vertices), False)
        axon_qual_label[sk.downstream_nodes(close_vind)] = True
        split_quality = axon_split_quality(axon_qual_label, pre_inds, post_inds)

    if extend_to_segment:
        relseg = sk.segment_map[close_vind]
        min_ind = np.argmin(sk.distance_to_root[sk.segments[relseg]])
        axon_split_ind = sk.segments[relseg][min_ind]
    else:
        axon_split_ind = close_vind

    if return_quality:
        return axon_split_ind, split_quality
    else:
        return axon_split_ind

        ############################
        # Topological branch order #
        ############################


def branch_order(nrn, return_as_skel=False):
    """Compute simple branch order, counting how many branch points are between a node and root.

    Parameters
    ----------
    nrn : Meshwork
        Meshwork with skeleton
    return_as_skel : bool, optional
        If True, returns an array for the skeleton instead of the mesh. Default is False.

    Returns
    -------
    branch_order
        N-length array containing the branch order for each mesh (or skeleton) vertex.
    """
    if nrn.skeleton is None:
        raise ValueError("Meshwork must have skeleton")

    has_bp = np.zeros(nrn.skeleton.n_vertices)
    has_bp[nrn.skeleton.branch_points] = 1

    covers = nrn.skeleton.cover_paths
    branch_order = np.zeros(nrn.skeleton.n_vertices)
    for cp in covers:
        path = cp[::-1]
        root_ind = nrn.skeleton.parent_nodes(path[0])
        if root_ind != -1:
            root_order = branch_order[root_ind]
        else:
            root_order = 0
        branch_order[path] = np.cumsum(has_bp[path]) + root_order
    branch_order[nrn.skeleton.root] = 0
    if return_as_skel:
        return branch_order
    else:
        return nrn.skeleton_property_to_mesh(branch_order, no_map_value=-1)


def _strahler_path(baseline):
    out = np.full(len(baseline), -1, dtype=np.int64)
    last_val = 1
    for ii in np.arange(len(out)):
        if baseline[ii] > last_val:
            last_val = baseline[ii]
        elif baseline[ii] == last_val:
            last_val += 1
        out[ii] = last_val
    return out


def strahler_order(nrn, return_as_skel=False):
    """Computes strahler number. Tips have strahler number 1, and strahler number increments
    when a two branches with the same strahler number merge.

    Parameters
    ----------
    nrn : Meshwork
        Meshwork neuron with skeleton
    return_as_skel : bool, optional
        Description

    Returns
    -------
    strahler_number : array
        Array of mesh (or, optionally, skeleton) vertices with strahler number of vertex.
    """
    if nrn.skeleton is None:
        raise ValueError("Meshwork must have skeleton")
    covers = nrn.skeleton.cover_paths
    strahler = np.full(nrn.skeleton.n_vertices, -1)
    for cp in covers[::-1]:
        new_vals = _strahler_path(strahler[cp])
        strahler[cp] = new_vals
        pind = nrn.skeleton.parent_nodes(cp[-1])
        if pind >= 0:
            if strahler[cp[-1]] > strahler[pind]:
                strahler[pind] = strahler[cp[-1]]
            elif strahler[cp[-1]] == strahler[pind]:
                strahler[pind] += 1
    if return_as_skel:
        return strahler
    else:
        return nrn.skeleton_property_to_mesh(strahler, no_map_value=-1)

    #####################
    # Density estimation #
    ######################


def _gaussian_kernel(x, sigma):
    def func(x):
        return np.exp(-x * x / (2 * sigma * sigma))

    return func


def _normalize_flat(W, nrn, exclude_root):
    if exclude_root:
        g = nrn.skeleton.cut_graph(
            nrn.skeleton.child_nodes([nrn.skeleton.root])[0], directed=False
        )
        len_per = np.array(g.sum(axis=1) / 2).ravel()
    else:
        len_per = np.array(nrn.skeleton.csgraph_undirected.sum(axis=1) / 2).ravel()
    return W.dot(len_per.reshape(-1, 1)).ravel()


def _normalize_gausssian(W):
    return W.sum(axis=1).squeeze()


def linear_density(
    nrn,
    inds,
    width,
    weight=None,
    kernel="flat",
    normalize=True,
    exclude_root=False,
):
    """Compute a sliding window average linear density of points across the object

    Parameters
    ----------
    inds : array
        Mesh indices for density (e.g. synapse locations).
    width : numeric
        width of average window (in all directions).
    weight : array None, optional
        Weight for each point for weighted average. If None, assumes weight of unity.
    normalize : bool, optional
        If False, sums the weights but does not normalize by amount of cable.
        Default is True.
    exclude_root : bool, optional
        If True, disconnects root from the graph for the case that the root is not
        well-approximated by a line (e.g. a cell body.). The density for those vertices
        will be infinite or nan.
        Default is False.


    Returns
    -------
    density_estimate : array
        N-length array of density at all mesh vertices.
    """
    if kernel == "flat":
        W = window_matrix(nrn.skeleton, width)
    elif kernel == "gaussian":
        dist_func = _gaussian_kernel(width)
        W = window_matrix(nrn.skeleton, 4 * width, dist_func)
    inds = nrn._convert_to_meshindex(inds)
    has_inds = np.full(nrn.skeleton.n_vertices, 0)
    if weight is None:
        skinds, count = np.unique(inds.to_skel_index, return_counts=True)
        has_inds[skinds] = count
    else:
        for w, skind in zip(weight, inds.to_skel_index):
            has_inds[skind] += w
    item_count = W.dot(has_inds.reshape(-1, 1)).ravel()
    if normalize:
        if kernel == "flat":
            norm = _normalize_flat(W, nrn, exclude_root)
        elif kernel == "gaussian":
            norm = _normalize_gausssian(W)

        with np.errstate(divide="ignore"):
            rho = item_count / norm
    else:
        rho = item_count
    return rho[nrn.skeleton.mesh_to_skel_map][nrn.mesh.node_mask]
