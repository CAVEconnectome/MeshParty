import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree as KDTree
from scipy.sparse import csgraph

DEFAULT_BINS_GRAPH = np.append(np.arange(0, 15), np.inf)
DEFAULT_P_RATIO = np.array([1.66337261,  1.47190522,  1.60522673,  1.27751097,  0.73914456,
                            -0.0630327, -1.44133147, -2.25994548, -2.77429373, -2.77863396,
                            -4.52941582, -4.68640886, -5.67060111, -6.41176085, -9.43203527])


def skeleton_path_quality(sk, mesh, skind_to_mind_map=None, sk_norm=None, max_graph_dist=10000,
                          p_ratio=None, bins=None, normalize_to_self=True,
                          sliding=True, window=400, interval=50, return_path_info=False, filter_skind_map=True):
    '''
    Compute path quality for all endpoints of a skeletonization by comparing paths from skeleton tips to root to
    the shortest path between the same points along the mesh.
    :param sk: MeshParty Skeleton, derived from the mesh also given as an argument. The skeleton has N_s vertices.
    :param mesh: MeshParty Mesh. The mesh has N_m vertices.
    :param skind_to_mind_map: N_s length array or list. Optional, default is None. Each the value of the ith
                entry is the index of the mesh vertex corresponding to the ith skeleton vertex. If none is given,
                the value defaults to sk.vertex_properties['mesh_index'], created by default in skeletonization.skeletonize_mesh.
    :param sk_norm: Float or N_s length array or list. Optional, default is None. A normalization value for the distance
                between paths to account for different-width neurites. If None is given, the value defaults to 
                sk.radius, the mesh radius value created by default in skeletonization.skeletonize_mesh.
                If the radius is used, the pre-computed default likelihood ratio/binning.
    :param max_graph_dist: Float. Optional, default=10000. Values behind this are treated as infinity in distance computations.
    :param p_ratio: List with M values. Optional, defualt is None. Denotes the log likelihood ratio that a given distance bin came from a matched
                path pair vs an unmatched path pair.  If None is given, it defaults to the value DEFAULT_P_RATIO above, trained on
                a collection of paths from seven pyramidal neurons in the Pinky100 dataset.
    :param bins: List with M+1 values. Optional, default is None. Gives the bin edges for the distances corresponding to the
                p_ratio values. If None is given, it defaults to DEFAULT_BINS_GRAPH.
    :param normalize_to_self: Boolean. Optional, default is True. If False, the sum of log likelihoods for path pairs is given.
                If True, the values are normalized to the best-case scenario of the log-likelihood distane of the skeleton path to itself.
    :param sliding: Boolean. Optional, default is True. If True, takes the worst score along a sliding window along the paths to avoid
                path length diluting a small misalignment between paths. If False, the whole paths are used.
    :param window: Int. Optional, default is 400. The path length (in hops) of the window to use for the sliding quality measure.
    :param interval: Int. Optional, default is 50. The interval between window starts in teh sliding quality measure.
    :param return_path_info: Boolean. Optional, default is False. If True, returns a list of path_quality values and the paths themselves.
                If False, returns only the path quality.
    :param filter_skind_map: Boolean. Optional, default is True. If True, assumes the skind_to_mind_map is in the indices of the original
                mesh, not a filtered version. Otherwise, ignores the node_mask. Should be True if using default values for skind_to_mind_map.
    :returns: path_quality (if return_path_info is False)
              path_quality, sk_paths, ms_paths, sk_inds_list, mesh_inds_list, path_distances (if return_path_info is True)
    '''
    if skind_to_mind_map is None:
        ds, inds = mesh.kdtree.query(sk.vertices, distance_upper_bound=1)
        skind_to_mind_map = mesh.map_indices_to_unmasked(
            np.where(ds == 0, inds, -1))
    if sk_norm is None:
        if sk.radius is not None:
            sk_norm = sk.radius
        else:
            sk_norm = np.ones(sk.n_vertices)
    elif p_ratio is not None:
        print(
            'Warning: If sk_norm is not the radius, the default calibration does not apply!')

    if p_ratio is None:
        p_ratio = DEFAULT_P_RATIO

    if bins is None:
        bins = DEFAULT_BINS_GRAPH

    if type(skind_to_mind_map) is list:
        skind_to_mind_map = np.array(skind_to_mind_map)
    if type(sk_norm) is list:
        sk_norm = np.array(sk_norm)
    if filter_skind_map is True:
        skind_to_mind_map = mesh.filter_unmasked_indices(skind_to_mind_map)
    print("Computing path pairs...")
    sk_paths, ms_paths, sk_inds_list, mesh_inds_list = compute_path_pairs(
        sk, mesh, skind_to_mind_map=skind_to_mind_map)
    print("Computing matched path distances...")
    path_distances = matched_path_distances_mesh(sk_inds_list, mesh_inds_list, skind_to_mind_map, mesh,
                                                 sk_norm=sk_norm, max_dist=max_graph_dist)
    path_quality = []
    for pd in path_distances:
        if sliding:
            path_quality.append(pblast_score_sliding(pd, p_ratio=p_ratio,
                                                     bins=bins, normalize_to_self=normalize_to_self))
        else:
            path_quality.append(pblast_score(pd, p_ratio=p_ratio, bins=bins,
                                             normalize_to_self=normalize_to_self))
    if return_path_info:
        return np.array(path_quality), sk_paths, ms_paths, sk_inds_list, mesh_inds_list, path_distances
    else:
        return np.array(path_quality)


def compute_path_pairs(sk, mesh, skind_to_mind_map=None, return_indices=True):
    '''
    For every end point on a skeleton, computes two shortest paths to root,
    one along the skeleton and one along the mesh.
    :param sk: Skeleton
    :param mesh: MeshParty Mesh
    :param skind_to_mind_map: Optional, Mapping from index in skeleton to index in the mesh.
                              Usually 'mesh_index' in skeleton vertex properties.
    :param return_indices: Optional, boolean. If included, also return indices, not just points.
    :returns: sk_paths ms_paths, sk_path_indices, ms_path_indices
    '''
    end_points = sk.end_points
    mesh_to_skel_map = sk.mesh_to_skel_map[mesh.node_mask]
    root_minds = np.flatnonzero(mesh_to_skel_map == sk.root)

    # Compute a minimal set of indices to use in the dijkstra to avoid extra computation if all soma indices are included.
    mesh_vec = np.zeros((len(mesh.vertices), 1))
    mesh_vec[root_minds] = 1
    meshgraph = mesh.csgraph > 0
    soma_boundary_plus_vec = (
        ((meshgraph * mesh_vec) > 0).astype(int) - mesh_vec).astype(int)
    soma_boundary_inds = np.flatnonzero(np.logical_and(
        (meshgraph * soma_boundary_plus_vec > 0), mesh_vec > 0))

    ds, Ps, _ = csgraph.dijkstra(mesh.csgraph, return_predecessors=True,
                                 indices=soma_boundary_inds, min_only=True)
    sk_paths = []
    ms_paths = []
    if return_indices:
        ms_path_indices = []
        sk_path_indices = []
    for sk_ep in tqdm(sk.end_points):
        rel_minds = np.flatnonzero(mesh_to_skel_map == sk_ep)
        if skind_to_mind_map is None:
            rel_minds = np.flatnonzero(mesh_to_skel_map == sk_ep)
            ms_ep = rel_minds[np.argmin(np.linalg.norm(
                mesh.vertices[rel_minds] - sk.vertices[sk_ep], axis=1))]
        else:
            ms_ep = skind_to_mind_map[sk_ep]
        ms_path_inds = []
        ms_ind = ms_ep
        while ms_ind != -9999:  # Set by csgraph.dijkstra
            ms_path_inds.append(ms_ind)
            ms_ind = Ps[ms_ind]

        # Don't include root, since the mesh path won't have it
        sk_path = sk.vertices[sk.path_to_root(sk_ep)[:-1]]
        # Don't include root, since the mesh path won't have it
        ms_path = mesh.vertices[ms_path_inds]
        if len(sk_path) > 0:
            sk_paths.append(sk_path)
            ms_paths.append(ms_path)
            if return_indices:
                sk_path_indices.append(np.array(sk.path_to_root(sk_ep)[:-1]))
                ms_path_indices.append(np.array(ms_path_inds))
    if return_indices:
        return sk_paths, ms_paths, sk_path_indices, ms_path_indices
    else:
        return sk_paths, ms_paths


def matched_path_distances_normalized(sk_paths, mesh_paths, sk_inds_list, sk_rs, r_min=50, from_path='skel'):
    '''
    Compute euclidean distances between all paired paths.
    '''
    path_distances = []
    for sk_path, mesh_path, sk_inds in zip(sk_paths, mesh_paths, sk_inds_list):
        if from_path == 'mesh':
            ds, c_skinds = KDTree(sk_path).query(mesh_path)
            ds_norm = ds / np.clip(sk_rs[c_skinds], r_min, None)
        elif from_path == 'skel':
            ds, c_minds = KDTree(mesh_path).query(sk_path)
            ds_norm = ds / np.clip(sk_rs[sk_inds], r_min, None)
        path_distances.append(ds_norm)
    return path_distances


def matched_path_distances_mesh(sk_inds_list, mesh_inds_list, skind_to_mind_map, mesh, sk_norm=None, max_dist=10000):
    '''
    Compute on-mesh distance from skeleton to mesh path between all paired paths.
    Optionally, normalize distance by a value for every skeleton point, typically the radius.
    '''
    path_distances = []
    for skind_path, mind_path in tqdm(zip(sk_inds_list, mesh_inds_list)):
        ds = closest_graph_distance_paths(skind_to_mind_map[skind_path], mind_path, mesh,
                                          norm=sk_norm[skind_path], max_dist=max_dist)
        path_distances.append(ds)
    return path_distances


def closest_graph_distance_paths(path_A, path_B, mesh, norm=None, norm_min=50, max_dist=10000):
    ds = csgraph.dijkstra(mesh.csgraph, indices=path_B,
                          min_only=True, limit=max_dist)
    if norm is None:
        return ds[path_A]
    else:
        return ds[path_A] / np.clip(norm, norm_min, None)


def pblast_score(data, p_ratio=DEFAULT_P_RATIO, bins=DEFAULT_BINS_GRAPH, normalize_to_self=True):
    '''
    For a set of path distances, compute a metric that is basically a naive Bayes likelihood ratio test
    between being from a 'good' skeletonization (positive numbers) and a path that uses a bad merge
    (negative numbers).
    '''
    d_ns, _ = np.histogram(data, bins=bins)
    if normalize_to_self is True:
        norm = np.sum(len(data) * p_ratio[0])
    else:
        norm = 1
    return np.sum(np.dot(p_ratio, d_ns)) / norm


def pblast_score_sliding(data, window=400, interval=50, p_ratio=DEFAULT_P_RATIO, bins=DEFAULT_BINS_GRAPH, normalize_to_self=True):
    if len(data) < window:
        return pblast_score(data, p_ratio=p_ratio, bins=bins, normalize_to_self=normalize_to_self)
    else:
        worst_pscore = np.inf
        for ii in np.arange(0, len(data)-window+interval, interval):
            new_pscore = pblast_score(data[ii:ii+window], p_ratio=p_ratio,
                                      bins=bins, normalize_to_self=normalize_to_self)
            if new_pscore < worst_pscore:
                worst_pscore = new_pscore
        return worst_pscore
