import numpy as np
from pykdtree import kdtree
from scipy.sparse import csgraph

DEFAULT_BINS_GRAPH = np.append(np.arange(0,15), np.inf)
DEFAULT_P_RATIO = np.array([1.66337261,  1.47190522,  1.60522673,  1.27751097,  0.73914456,
                            -0.0630327 , -1.44133147, -2.25994548, -2.77429373, -2.77863396,
                            -4.52941582, -4.68640886, -5.67060111, -6.41176085, -9.43203527])

def skeleton_path_quality(sk, mesh, skind_to_mind_map, sk_norm, max_graph_dist=10000,
                          p_ratio=DEFAULT_P_RATIO, bins=DEFAULT_BINS_GRAPH, normalize_to_self=True,
                          sliding=True, window=400, interval=50, return_path_info=False):
    '''
    Compute path quality for all endpoints of a skeletonization.
    '''
    if type(skind_to_mind_map) is list:
        skind_to_mind_map = np.array(skind_to_mind_map)
    if type(sk_norm) is list:
        sk_norm = np.array(sk_norm)
    sk_paths, ms_paths, sk_inds_list, mesh_inds_list = compute_path_pairs(sk, mesh, skind_to_mind_map=skind_to_mind_map)
    path_distances = matched_path_distances_mesh(sk_inds_list, mesh_inds_list, skind_to_mind_map, mesh,
                                                 sk_norm=sk_norm, max_dist=max_graph_dist)
    path_quality = []
    for pd in path_distances:
        if sliding:
            path_quality.append(pblast_score_sliding(pd, p_ratio=p_ratio, bins=bins, normalize_to_self=normalize_to_self))
        else:
            path_quality.append(pblast_score(pd, p_ratio=p_ratio, bins=bins, normalize_to_self=normalize_to_self))
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
    soma_minds = np.flatnonzero(sk.mesh_to_skel_map==sk.root)

    # Compute a minimal set of indices to use in the dijkstra to avoid extra computation if all soma indices are included.
    mesh_vec = np.zeros((len(mesh.vertices),1))
    mesh_vec[soma_minds] = 1
    meshgraph = mesh.csgraph>0
    soma_boundary_plus_vec = (((meshgraph * mesh_vec)>0).astype(int) - mesh_vec).astype(int)
    soma_boundary_inds = np.flatnonzero(np.logical_and((meshgraph * soma_boundary_plus_vec > 0), mesh_vec>0))
    
    ds, Ps, _ = csgraph.dijkstra(mesh.csgraph, return_predecessors=True, indices=soma_boundary_inds, min_only=True)
    sk_paths = []
    ms_paths = []
    if return_indices:
        ms_path_indices = []
        sk_path_indices = []
    for sk_ep in sk.end_points:
        rel_minds = np.flatnonzero(sk.mesh_to_skel_map==sk_ep)
        if skind_to_mind_map is None:
            rel_minds = np.flatnonzero(sk.mesh_to_skel_map==sk_ep)
            ms_ep = rel_minds[np.argmin(np.linalg.norm(mesh.vertices[rel_minds] - sk.vertices[sk_ep], axis=1))]
        else:
            ms_ep = skind_to_mind_map[sk_ep]
        ms_path_inds = []
        ms_ind = ms_ep
        while ms_ind != -9999:
            ms_path_inds.append(ms_ind)
            ms_ind = Ps[ms_ind]
            
        sk_path = sk.vertices[sk.path_to_root(sk_ep)[:-1]]   #Don't include root, since the mesh path won't have it
        ms_path = mesh.vertices[ms_path_inds]   #Don't include root, since the mesh path won't have it

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
            ds, c_skinds  = kdtree.KDTree(sk_path).query(mesh_path)
            ds_norm = ds / np.clip(sk_rs[c_skinds], r_min, None)
        elif from_path == 'skel':
            ds, c_minds = kdtree.KDTree(mesh_path).query(sk_path)
            ds_norm = ds / np.clip(sk_rs[sk_inds], r_min, None)
        path_distances.append(ds_norm)
    return path_distances

def matched_path_distances_mesh(sk_inds_list, mesh_inds_list, skind_to_mind_map, mesh, sk_norm=None, max_dist=10000):
    '''
    Compute on-mesh distance from skeleton to mesh path between all paired paths.
    Optionally, normalize distance by a value for every skeleton point, typically the radius.
    '''
    path_distances = []
    for skind_path, mind_path in zip(sk_inds_list, mesh_inds_list):
        ds = closest_graph_distance_paths(skind_to_mind_map[skind_path], mind_path, mesh,
                                          norm=sk_norm[skind_path], max_dist=max_dist)
        path_distances.append(ds)
    return path_distances

def closest_graph_distance_paths(path_A, path_B, mesh, norm=None, norm_min=50, max_dist=10000):
    ds = csgraph.dijkstra(mesh.csgraph, indices=path_B, min_only=True, limit=max_dist)
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


def find_mixed_segments(pscore, sk, pscore_fail=0):
    is_good_score = np.zeros(len(sk.vertices))
    is_good_pot = np.zeros(len(sk.vertices))

    for ii, ep in enumerate(sk.end_points):
        ptr = sk.path_to_root(ep)
        is_good_pot[ptr] += 1
        if pscore[ii] > pscore_fail:
            is_good_score[ptr] += 1

    mixed_segs = []
    for jj, seg in enumerate(sk.segments):
        if len(seg)>1:
            end_bad = is_good_score[seg[0]]==0
            start_ind = sk.parent_node(seg[-1])
            start_mixed = is_good_score[start_ind] > 0 and is_good_score[start_ind] < is_good_pot[start_ind]
            if end_bad and start_mixed:
                mixed_segs.append(jj)

    return mixed_segs