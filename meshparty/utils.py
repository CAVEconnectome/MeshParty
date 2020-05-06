import numpy as np
from scipy import sparse
import networkx as nx


def connected_component_slice(G, ind=None, return_boolean=False):
    '''
    Gets a numpy slice of the connected component corresponding to a
    given index. If no index is specified, the slice is of the largest
    connected component.
    '''
    _, labels = sparse.csgraph.connected_components(G)
    if ind is None:
        label_vals, cnt = np.unique(labels, return_counts=True)
        ind = np.argmax(cnt)
        label = label_vals[ind]
    else:
        label = labels[ind]

    if return_boolean:
        return labels == label
    else:
        return np.flatnonzero(labels == label)


def dist_from_line(pts, line_bound_pts, axis):
    ps = (pts[:, axis] - line_bound_pts[0, axis]) / \
        (line_bound_pts[1, axis] - line_bound_pts[0, axis])
    line_pts = np.multiply(ps[:, np.newaxis], line_bound_pts[1] -
                           line_bound_pts[0]) + line_bound_pts[0]
    ds = np.linalg.norm(pts - line_pts, axis=1)
    return ds


def filter_close_to_line(mesh, line_bound_pts, line_dist_th, axis=1):
    line_pt_ord = np.argsort(line_bound_pts[:, axis])
    below_top = mesh.vertices[:, axis] > line_bound_pts[line_pt_ord[0], axis]
    above_bot = mesh.vertices[:, axis] < line_bound_pts[line_pt_ord[1], axis]
    ds = dist_from_line(mesh.vertices, line_bound_pts, axis)
    is_close = (ds < line_dist_th) & below_top & above_bot
    return is_close


def mutual_closest_edges(mesh_a, mesh_b, distance_upper_bound=250):
    _, a_inds = mesh_a.kdtree.query(mesh_b.vertices,
                                    distance_upper_bound=distance_upper_bound)
    b_ds, b_inds = mesh_b.kdtree.query(mesh_a.vertices,
                                       distance_upper_bound=distance_upper_bound)
    mutual_closest = b_inds[a_inds[b_inds[~np.isinf(b_ds)]]] == b_inds[~np.isinf(b_ds)]

    a_closest = a_inds[b_inds[~np.isinf(b_ds)]][mutual_closest]
    b_closest = b_inds[~np.isinf(b_ds)][mutual_closest]
    potential_edges = np.unique(np.vstack((a_closest, b_closest)), axis=1).T
    return potential_edges[:, 0], potential_edges[:, 1]


def indices_to_slice(inds, total_length):
    v = np.full(total_length, False)
    v[inds] = True
    return v


def find_far_points(trimesh, start_ind=None, multicomponent=False):
    '''
    Wraps find_far_points_graph for meshes instead of csgraphs.
    '''
    return find_far_points_graph(trimesh.csgraph,
                                 start_ind=start_ind,
                                 multicomponent=multicomponent)


def find_far_points_graph(mesh_graph, start_ind=None, multicomponent=False):
    '''
    Finds the maximally far point along a graph by bouncing from farthest point
    to farthest point.
    '''
    d = 0
    dn = 1

    if start_ind is None:
        if multicomponent:
            a = connected_component_slice(mesh_graph)[0]
        else:
            a = 0
    else:
        a = start_ind
    b = 1

    k = 0
    pred = None
    ds = None
    while 1:
        k += 1
        dsn, predn = sparse.csgraph.dijkstra(
            mesh_graph, False, a, return_predecessors=True)
        if multicomponent:
            dsn[np.isinf(dsn)] = 0
        bn = np.argmax(dsn)
        dn = dsn[bn]
        if dn > d:
            b = a
            a = bn
            d = dn
            pred = predn
            ds = dsn
        else:
            break

    return b, a, pred, d, ds


def edge_averaged_vertex_property(edge_property, vertices, edges):
    '''
    Converts a per-edge property to a vertex property by taking the mean
    of the adjacent edges.
    '''
    vertex_property = np.full((len(vertices), 2), np.nan)
    vertex_property[edges[:, 0], 0] = np.array(edge_property)
    vertex_property[edges[:, 1], 1] = np.array(edge_property)
    return np.nanmean(vertex_property, axis=1)


def reduce_vertices(vertices, vertex_shape, v_filter=None, e_filter=None, return_filter_inds=False):
    '''
    Given a vertex and edge list, filters them down and remaps indices in the edge list.
    If no v or e filters are given, reduces the vertex list down to only those vertices
    with values in the vertex_shape object.
    '''
    if v_filter is None:
        v_filter = np.unique(vertex_shape).astype(int)
    if v_filter.dtype == bool:
        v_filter = np.flatnonzero(v_filter)
    if e_filter is None:
        # Take all edges that have both vertices in the kept indices
        e_filter_bool = np.all(np.isin(vertex_shape, v_filter), axis=1)
        e_filter = np.flatnonzero(e_filter_bool)

    vertices_n = vertices[v_filter]
    vertex_shape_n = filter_shapes(v_filter, vertex_shape[e_filter])

    if return_filter_inds:
        return vertices_n, vertex_shape_n, (v_filter, e_filter)
    else:
        return vertices_n, vertex_shape_n


def create_csgraph(vertices, edges, euclidean_weight=True, directed=False):
    '''
    Builds a csr graph from vertices and edges, with optional control
    over weights as boolean or based on Euclidean distance.
    '''
    if euclidean_weight:
        xs = vertices[edges[:, 0]]
        ys = vertices[edges[:, 1]]
        weights = np.linalg.norm(xs-ys, axis=1)
        use_dtype = np.float32
    else:
        weights = np.ones((len(edges),)).astype(np.int8)
        use_dtype = np.int8

    if directed:
        edges = edges.T
    else:
        edges = np.concatenate([edges.T, edges.T[[1, 0]]], axis=1)
        weights = np.concatenate([weights, weights]).astype(dtype=use_dtype)

    csgraph = sparse.csr_matrix((weights, edges),
                                shape=[len(vertices), ] * 2,
                                dtype=use_dtype)

    return csgraph


def create_nxgraph(vertices, edges, euclidean_weight=True, directed=False):
    if euclidean_weight:
        xs = vertices[edges[:, 0]]
        ys = vertices[edges[:, 1]]
        weights = np.linalg.norm(xs-ys, axis=1)
        use_dtype = np.float32
    else:
        weights = np.ones((len(edges),)).astype(bool)
        use_dtype = bool

    if directed:
        edges = edges.T
    else:
        edges = np.concatenate([edges.T, edges.T[[1, 0]]], axis=1)
        weights = np.concatenate([weights, weights]).astype(dtype=use_dtype)

    weighted_graph = nx.Graph()
    weighted_graph.add_edges_from(edges)

    for i_edge, edge in enumerate(edges):
        weighted_graph[edge[0]][edge[1]]['weight'] = weights[i_edge]
        weighted_graph[edge[1]][edge[0]]['weight'] = weights[i_edge]

    return weighted_graph


def get_path(root, target, pred):
    '''
    Using a predecessor matrix from scipy.csgraph.shortest_paths, get all indices
    on the path from a root node to a target node.
    '''
    path = [target]
    p = target
    while p != root:
        p = pred[p]
        path.append(p)
    path.reverse()
    return path


def paths_to_edges(path_list):
    '''
    Turn an ordered path into an edge list.
    '''
    arrays = []
    for path in path_list:
        p = np.array(path)
        e = np.vstack((p[0:-1], p[1:])).T
        arrays.append(e)
    return np.vstack(arrays)


def filter_shapes(node_ids, shapes):
    """ node_ids has to be sorted! """
    if not isinstance(node_ids[0], list) and \
            not isinstance(node_ids[0], np.ndarray):
        node_ids = [node_ids]
    # If shapes is 1d, make into an Nx1 2d-array.
    if len(shapes.shape) == 1:
        shapes = shapes[:, np.newaxis]
    ndim = shapes.shape[1]
    if isinstance(node_ids, np.ndarray):
        all_node_ids = node_ids.flatten()
    else:
        all_node_ids = np.concatenate(node_ids)

    filter_ = np.in1d(shapes[:, 0], all_node_ids)
    pre_filtered_shapes = shapes[filter_].copy()
    for k in range(1, ndim):
        filter_ = np.in1d(pre_filtered_shapes[:, k], all_node_ids)
        pre_filtered_shapes = pre_filtered_shapes[filter_]

    filtered_shapes = []

    for ns in node_ids:
        f = pre_filtered_shapes[np.in1d(pre_filtered_shapes[:, 0], ns)]
        for k in range(1, ndim):
            f = f[np.in1d(f[:, k], ns)]

        f = np.unique(np.concatenate([f.flatten(), ns]),
                      return_inverse=True)[1][:-len(ns)].reshape(-1, ndim)

        filtered_shapes.append(f)

    return filtered_shapes


def nanfilter_shapes(node_ids, shapes):
    '''
    Wraps filter_shapes to handle shapes with nans.
    '''
    # if not any(np.isnan(shapes)):
    #     return filter_shapes(node_ids, shapes)[0].reshape(shapes.shape)

    long_shapes = shapes.ravel()
    ind_rows = ~np.isnan(long_shapes)
    new_inds = filter_shapes(node_ids, long_shapes[ind_rows])

    filtered_shape = np.full(len(long_shapes), np.nan)
    filtered_shape[ind_rows] = new_inds[0].ravel()
    return filtered_shape.reshape(shapes.shape)


def path_from_predecessors(Ps, ind_start):
    """
    Build a path from an initial index to a target node based
    on the target node predecessor row from a shortest path query.
    """
    path = []
    next_ind = ind_start
    while next_ind != -9999:
        path.append(next_ind)
        next_ind = Ps[next_ind]
    return np.array(path)


def map_indices_to_unmasked(indices_unmasked, unmapped_indices):
    '''
    For a set of masked indices, returns the corresponding unmasked indices

    Parameters
    ----------
    unmapped_indices: np.array
        a set of indices in the masked index space

    Returns
    -------
    np.array
        the indices mapped back to the original mesh index space
    '''
    return indices_unmasked[unmapped_indices]


def map_boolean_to_unmasked(unmasked_size, node_mask, unmapped_boolean):
    '''
    For a boolean index in the masked indices, returns the corresponding unmasked boolean index

    Parameters
    ----------
    unmapped_boolean : np.array
        a bool array in the masked index space

    Returns
    -------
    np.array
        a bool array in the original index space.  Is True if the unmapped_boolean suggests it should be.
    '''
    full_boolean = np.full(unmasked_size, False)
    full_boolean[node_mask] = unmapped_boolean
    return full_boolean


def filter_unmasked_boolean(node_mask, unmasked_boolean):
    '''
    For an unmasked boolean slice, returns a boolean slice filtered to the masked mesh

    Parameters
    ----------
    unmasked_boolean : np.array
        a bool array in the original mesh index space

    Returns
    -------
    np.array
        returns the elements of unmasked_boolean that are still relevant in the masked index space
    '''
    return unmasked_boolean[node_mask]


def filter_unmasked_indices(node_mask, unmasked_shape):
    """
    filters a set of indices in the original mesh space
    and returns it in the masked space

    Parameters
    ----------
    mask: np.array or None
        the mask to apply. default None will use this Mesh node_mask

    unmasked_shape: np.array
        a set of indices into vertices in the unmasked index space

    Returns
    -------
    np.array
        the unmasked_shape indices mapped into the masked index space
    """
    new_shape = filter_unmasked_indices_padded(node_mask, unmasked_shape)

    if len(new_shape.shape) > 1:
        keep_rows = np.all(new_shape >= 0, axis=1)
    else:
        keep_rows = new_shape >= 0

    return new_shape[keep_rows]


def filter_unmasked_indices_padded(node_mask, unmasked_shape):
    new_index = np.zeros(node_mask.shape)-1
    new_index[node_mask] = np.arange(np.sum(node_mask))

    if np.isscalar(unmasked_shape) is True:
        new_shape = int(new_index[unmasked_shape])
    else:
        unmasked_shape = np.array(unmasked_shape)
        new_shape = new_index[unmasked_shape.ravel()].reshape(unmasked_shape.shape).astype(int)
    return new_shape
