import numpy as np
import numba

#####################
# Split by synapses #
#####################


def split_axon_by_synapses(
    nrn, pre_inds, post_inds, return_quality=True, extend_to_segment=True
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

    axon_split = _find_axon_split(
        nrn.skeleton,
        pre_inds,
        post_inds,
        return_quality=return_quality,
        extend_to_segment=extend_to_segment,
    )

    if return_quality:
        axon_split_ind, split_quality = axon_split
    else:
        axon_split_ind = axon_split
    is_axon_sk = np.full(len(sk.vertices), False)
    is_axon_sk[sk.downstream_nodes(axon_split_ind)] = True

    is_axon = nrn.SkeletonIndex(np.flatnonzero(is_axon_sk)).to_mesh_index

    if return_quality:
        return is_axon, split_quality
    else:
        return is_axon


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


def _synapse_betweenness(sk, pre_inds, post_inds, use_entropy=False):

    Npre, n_pre = _precompute_synapse_inds(sk, pre_inds)
    Npost, n_post = _precompute_synapse_inds(sk, post_inds)

    syn_btwn = np.zeros(len(sk.vertices))
    split_index = np.zeros(len(sk.vertices))
    cov_paths_rev = sk.cover_paths[::-1]
    if use_entropy:
        entropy_normalization = distribution_split_entropy(
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
                    1 - distribution_split_entropy(counts) / entropy_normalization
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
    syn_btwn = _synapse_betweenness(sk, pre_inds, post_inds)
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
