import numpy as numpy
from scipy import sparse


def connected_component_slice(G, ind=None):
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
    return np.where(labels == label)[0]


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
    

def reduce_vertices(vertices, edges, v_filter=None, e_filter=None, return_filter_inds=False):
    '''
    Given a vertex and edge list, filters them down and remaps indices in the edge list.
    If no v or e filters are given, reduces the vertex list down to only those vertices
    with edges in the edge list.
    '''
    if v_filter is None:
        v_filter = np.unique(edges).astype(int)
    if e_filter is None:
        e_filter_bool = np.isin(edges[:,0], v_filter) & np.isin(edges[:,1], v_filter)
        e_filter = np.where(e_filter_bool)[0]
    
    vertices_n = vertices[v_filter]
    vmap = dict(zip(v_filter, np.arange(len(v_filter))))
    
    edges_f = edges[e_filter].copy()
    edges_n = np.stack((np.fromiter((vmap[x] for x in edges_f[:,0]), dtype=int),
                        np.fromiter((vmap[x] for x in edges_f[:,1]), dtype=int))).T

    if return_filter_inds:
        return vertices_n, edges_n, (v_filter, e_filter)
    else:
        return vertice_n, edges_n


def create_csgraph(vertices,
                   edges,
                   euclidean_weight=True):
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

    edges = np.concatenate([edges.T, edges.T[[1, 0]]], axis=1)
    weights = np.concatenate([weights, weights]).astype(dtype=use_dtype)

    csgraph = sparse.csr_matrix((weights, edges),
                                shape=[len(vertices), ] * 2,
                                dtype=use_dtype)

    return csgraph


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
