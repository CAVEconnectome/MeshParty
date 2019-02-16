import numpy as np 
from scipy import sparse
import networkx as nx
import pcst_fast


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

def indices_to_slice(inds, total_length):
    v = np.full(total_lenth, False)
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
    vertex_property = np.full((len(vertices),2), np.nan)
    vertex_property[edges[:,0],0] = np.array(edge_property)
    vertex_property[edges[:,1],1] = np.array(edge_property)
    return np.nanmean(vertex_property, axis=1)
    

def reduce_vertices(vertices, vertex_list, v_filter=None, e_filter=None, return_filter_inds=False):
    '''
    Given a vertex and edge list, filters them down and remaps indices in the edge list.
    If no v or e filters are given, reduces the vertex list down to only those vertices
    with edges in the edge list.
    '''
    if v_filter is None:
        v_filter = np.unique(vertex_list).astype(int)
    if v_filter.dtype == bool:
        v_filter = np.flatnonzero(v_filter)
    if e_filter is None:
        e_filter_bool = np.all(np.isin(vertex_list, v_filter), axis=1)
        e_filter = np.flatnonzero(e_filter_bool)

    vertices_n = vertices[v_filter]
    vmap = dict(zip(v_filter, np.arange(len(v_filter))))
    
    # Remap to the reduced vertex indices
    vertex_list_f = vertex_list[e_filter]
    remapped_col = []
    for i in range(np.shape(vertex_list_f)[1]):
        remapped_col.append(np.fromiter((vmap[x] for x in vertex_list_f[:,i]), dtype=int))
    vertex_list_n = np.stack(remapped_col).T

    if return_filter_inds:
        return vertices_n, vertex_list_n, (v_filter, e_filter)
    else:
        return vertices_n, vertex_list_n


def create_csgraph(vertices, edges, euclidean_weight=True, directed=False):
    '''
    Builds a csr graph from vertices and edges, with optional control
    over weights as boolean or based on Euclidean distance.
    '''
    if euclidean_weight:
        xs = vertices[edges[:,0]]
        ys = vertices[edges[:,1]]
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

    csgraph = sparse.csr_matrix((weights, edges),
                                shape=[len(vertices), ] * 2,
                                dtype=use_dtype)

    return csgraph


def create_nxgraph(vertices, edges, euclidean_weight=True, directed=False):
    if euclidean_weight:
        xs = vertices[edges[:,0]]
        ys = vertices[edges[:,1]]
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


def get_steiner_mst(trimesh, positions, d_upper_bound=1000):
    """ calculates the steiner mst tree that visits the mesh nodes closest
        to a set of positions.

        :param trimesh: meshparty.trimesh_io.Mesh
        :param positions: a Mx3 numpy array of positions to visit

        :return: verts, edges
        verts is a Nx3 list of vertices of the graph
        edges is a Kx2 list of edges of the tree
    """
    dists, node_ids = trimesh.kdtree.query(positions, 1,
                                           distance_upper_bound=d_upper_bound,
                                           n_jobs=-1)
    # calculate the weights as the distance along the edges
    weights = np.float64(np.linalg.norm(trimesh.vertices[trimesh.edges[:, 0]] -
                                        trimesh.vertices[trimesh.edges[:, 1]],
                                        axis=1))
    # set the prizes equal to all the weights,
    # so its always worth visiting a prized node if possible
    prizes = np.zeros(len(trimesh.vertices))
    prizes[node_ids] = np.sum(weights)

    # use pcst fast to calculate the steiner tree approximation
    mst_verts, mst_edges = pcst_fast.pcst_fast(
        trimesh.edges, prizes, weights, -1, 1, 'gw', 1)

    # pcst fast answer is in terms of indices of its inputs
    new_edges = trimesh.edges[mst_edges, :]
    # pull out the selected vertices
    mst_verts_red = trimesh.vertices[mst_verts, :]
    # we need to reindex the edges into the new list of vertices
    mst_edges_red = np.zeros(new_edges.shape, dtype=new_edges.dtype)
    for i in range(2):
        mst_edges_red[:, i] = np.searchsorted(mst_verts, new_edges[:, i])
    return mst_verts_red, mst_edges_red


def filter_shapes(node_ids, shapes):
    """ node_ids has to be sorted! """
    if not isinstance(node_ids[0], list) and \
            not isinstance(node_ids[0], np.ndarray):
        node_ids = [node_ids]
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
