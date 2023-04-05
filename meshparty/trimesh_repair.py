
from scipy import spatial, sparse
from meshparty import utils
import time
import numpy as np
from meshparty import trimesh_io
import logging
try:
    from caveclient import CAVEclient
except ImportError:
    logging.warning(
        "Need to pip install caveclient to repair mesh with pychunkedgraph")


def np_shared_rows(A, B):
    """ a utility to find intersection of two arrays

    Parameters
    ----------
    A: np.array
        a Nx2 array of np.int32 values
    B: np.array
        a Mx2 array of np.int32 values

    Returns
    -------
    np.array
        a Kx2 array of np.int32 values where the rows of A were also in B
    """
    A_flat = np.ascontiguousarray(np.sort(A, axis=1)).view(dtype=('i8,i8'))
    B_flat = np.ascontiguousarray(np.sort(B, axis=1)).view(dtype=('i8,i8'))
    good_edges = np.where(np.in1d(A_flat, B_flat))[0]
    return good_edges


def find_close_edges(vertices, labels, label_a, label_b):
    """ a function to find the closest vertex in one compoment for every vertex of another
    component

    Parameters
    ----------
    vertices: np.array
        a Nx3 array of mesh vertex locations
    labels: np.array
        A N np.array of ints across vertices where connected components have the same label
    label_a: int
        the label of the first connected component
    label_b: int
        the label of the second connected component

    Returns
    -------
    np.array
        close_edges, a Kx2 array (where K is the np.sum(labels==label_a) of vertex indices,
        where the first column entry is all the vertices in  component a
        and the second column entry is the index of the vertex in component b that is closest
        to the vertex in the first column.
    """
    a_inds = np.where(labels == label_a)[0]
    b_inds = np.where(labels == label_b)[0]
    b_tree = spatial.cKDTree(vertices[b_inds, :], balanced_tree=False)
    ds, closest = b_tree.query(vertices[a_inds, :], k=1)
    b_close = b_inds[closest[~np.isinf(ds)]]
    a_close = a_inds[~np.isinf(ds)]
    return np.hstack((a_close[:, np.newaxis], b_close[:, np.newaxis]))


def find_close_edges_sym(vertices, labels, label_a, label_b):
    """ a function to find the mutually closest vertices between two compoments of a mesh
    meaning.. vertex 1 is the closest vertex from component a to vertex 2
    and vertex 2 is the closest vertex in component b to vertex 1.

    Parameters
    ----------
    vertices: np.array
        a Nx3 array of mesh vertex locations
    labels: np.array
        A N np.array of ints across vertices where connected components have the same label
    label_a: int
        the label of the first connected component
    label_b: int
        the label of the second connected component

    Returns
    -------
    np.array
        close_edges, a Kx2 array of vertex indices,
        where the first column entry is an index in component a
        the second column entry is an index in component b
        and each vertex is mutually the closest vertex to the other component
    """

    new_edges_a = find_close_edges(vertices,
                                   labels,
                                   label_a,
                                   label_b)
    new_edges_b = find_close_edges(vertices,
                                   labels,
                                   label_b,
                                   label_a)
    is_both = np_shared_rows(new_edges_a, new_edges_b)
    return new_edges_a[is_both, :]


def find_all_close_edges(vertices, labels, ccs):
    """ Find all the mutually closest edges between all components of the mesh

    Parameters
    ----------
    vertices: np.array
        a Nx3 array of xyz vertex position
    labels: np.array
        a labelling of the vertices that reflect components of the mesh
    ccs: int
        the number of connected components in the mesh

    Returns
    -------
    np.array
        a Kx2 array of vertex indices that are mutually closest to each other
        across all combinations of components
    """
    all_edges = []
    for i in range(ccs):
        for j in range(i, ccs):
            all_edges.append(find_close_edges_sym(vertices, labels, i, j))
    return np.vstack(all_edges)


def find_edges_to_link(mesh, vert_ind_a, vert_ind_b, distance_upper_bound=2500, verbose=False):
    '''Given a mesh and two points on that mesh
    find a way to add edges to the  mesh graph so that those indices
    are on the same connected component

    Parameters
    ----------
    mesh: trimesh_io.Mesh
        a mesh to find edges on
    vert_ind_a: int
        one index into mesh.vertices, the first point
    vert_ind_b: int
        a second index into mesh.vertices, the second point
    distance_upper_bound: float
        a maximum distance to (default 2500 in units of mesh.vertices)
    verbose: bool
        whether to print debug info

    Returns
    -------
    np.array
        a Kx2 array of mesh indices that represent edges to add to the mesh to link the two points
        in a way that creates the shortest path between the points across mututally closest vertices
        from connected components.. not adding edges if they are larger than distance_upper_bound
        TODO: distance_upper_bound not presently implemented
    '''
    timings = {}
    start_time = time.time()

    # find the distance between the merge points and their center
    d = np.linalg.norm(
        mesh.vertices[vert_ind_a, :] - mesh.vertices[vert_ind_b, :])
    c = np.mean(mesh.vertices[[vert_ind_a, vert_ind_b], :], axis=0)
    # cut down the mesh to only include mesh vertices near the center of this
    # merge edge and within 2x the euclidean length of the edge
    inds = mesh.kdtree.query_ball_point(c, d*2)
    # convert this to a mask
    mask = np.zeros(len(mesh.vertices), dtype=bool)
    mask[inds] = True

    timings['create_mask'] = time.time()-start_time
    start_time = time.time()
    # create a masked version of the mesh
    mask_mesh = mesh.apply_mask(mask)

    timings['apply_mask'] = time.time()-start_time
    start_time = time.time()
    ccs, labels = sparse.csgraph.connected_components(
        mask_mesh.csgraph, return_labels=True)

    # map the original indices into this masked space
    mask_inds = mask_mesh.filter_unmasked_indices(
        np.array([vert_ind_a, vert_ind_b]))

    timings['masked_ccs'] = time.time()-start_time
    start_time = time.time()

    # find all the multually closest edges between the linked components
    new_edges = find_close_edges_sym(mask_mesh.vertices,
                                     labels,
                                     labels[mask_inds[0]],
                                     labels[mask_inds[1]])
    timings['find_close_edges_sym'] = time.time()-start_time
    start_time = time.time()

    # if there is now way to do this, fall back to adding all
    # edges that are close
    if len(new_edges) == 0:
        if verbose:
            print('finding all close edges')
        new_edges = find_all_close_edges(mask_mesh.vertices, labels, ccs)
        if verbose:
            print(f'new_edges shape {new_edges.shape}')
    # if there are still not edges we have a problem
    if len(new_edges) == 0:
        raise Exception('no close edges found')

    # create a new mesh that has these added edges
    # new_mesh = make_new_mesh_with_added_edges(mask_mesh, new_edges)
    total_edges = np.vstack([mask_mesh.graph_edges, new_edges])
    graph = utils.create_csgraph(mask_mesh.vertices, total_edges)
    timings['make_new_mesh'] = time.time()-start_time
    start_time = time.time()

    # find the shortest path to one of the linking spots in this new mesh
    d_ais_to_all, pred = sparse.csgraph.dijkstra(graph,
                                                 indices=mask_inds[0],
                                                 unweighted=False,
                                                 directed=False,
                                                 return_predecessors=True)
    timings['find_close_edges_sym'] = time.time()-start_time
    start_time = time.time()
    # make sure we found a good path
    if np.isinf(d_ais_to_all[mask_inds[1]]):
        raise Exception(
            f"cannot find link between {vert_ind_a} and {vert_ind_b}")

    # turn this path back into a original mesh index edge list
    path = utils.get_path(mask_inds[0], mask_inds[1], pred)
    path_as_edges = utils.paths_to_edges([path])
    good_edges = np_shared_rows(path_as_edges, new_edges)
    good_edges = np.sort(path_as_edges[good_edges], axis=1)
    timings['remap answers'] = time.time()-start_time
    if verbose:
        print(timings)
    return mask_mesh.map_indices_to_unmasked(good_edges)


def merge_points_to_merge_indices(mesh, merge_event_points, close_map_distance=300):
    """ function to take merge points in 3d space and find the right mesh.vertices indices
    filtering out merge events that merge the same component or are far away

    Parameters
    ----------
    mesh : trimesh_io.Mesh
        mesh to map points to
    merge_event_points: np.array
        a Nx2x3 matrix of merge points, where the last dimension is xyz
    close_map_distance: float or int
        a maximum distance to map points to indices (default 300)

    Returns
    -------
    np.array
        close_inds, a Mx2 matrix of indices into mesh.vertices
        M<N because some merge pairs might not have been able to both be mapped
        in under close_map_distance, and merge_points that only seem to connect
        already connected components are not added.
    """
    Nmerge = merge_event_points.shape[0]

    # find the connected components of the mesh
    ccs, labels = sparse.csgraph.connected_components(
        mesh.csgraph, return_labels=True)
    uniq_labels, label_counts = np.unique(labels, return_counts=True)
    large_cc_mask = np.isin(labels, uniq_labels[label_counts > 100])

    # convert the edge list to a Nmergex3 list of vertex positions
    merge_edge_verts = merge_event_points.reshape((merge_event_points.shape[0]*merge_event_points.shape[1],
                                                   merge_event_points.shape[2]))
    # create a fake list of indices for a skeleton
    # merge_edge_inds = np.arange(0, len(merge_edge_verts)).reshape(Nmerge, 2)

    # make a kdtree of the mesh triangle centers
    tree = spatial.cKDTree(mesh.triangles_center, balanced_tree=False)
    # find the close triangle centers to the merge vertices
    ds2, close_face_inds = tree.query(merge_edge_verts)
    # pick one of the triangle faces as our close index
    close_inds = mesh.faces[close_face_inds, 0]

    # lookup what connected component each close index is
    # reshaping it into a Nmerge x 2 format
    cc_labels = labels[close_inds].reshape((Nmerge, 2))

    # we want to differeniate merges that connected different
    # connected components from those that are connected disconnected
    # components of the mesh
    is_join_merge = cc_labels[:, 0] != cc_labels[:, 1]

    # however we can't trust that our mapping step worked perfectly
    # as euclidean distance can be tricky... so we do a second check
    # on the merge edges that end up on the same connected component
    # to see if there is another close vertex from a different connected
    # component

    # these are the merge edges that had 'close' results on at least one
    # of their edges
    close_mapped = np.min(ds2.reshape((Nmerge, 2)),
                          axis=1) < close_map_distance
    # calculate the index of merge edges that are mapped closely,
    # but ended up being on the same connected component
    # these are remapping candidates
    remap_merges = np.where(close_mapped & ~is_join_merge)[0]
    if (np.sum(remap_merges) == 0):
        return close_inds.reshape((Nmerge, 2))[is_join_merge, :]

    # pick out the coordinates of our candidates
    merge_coords = merge_event_points[close_mapped, :, :]
    # figure out which of the endpoint of the merge edge was the closest one
    close_merge = np.argmin(ds2.reshape((Nmerge, 2)), axis=1)

    # we want to reconsider the other end of the merge edge and get their xyz coords
    far_points = merge_event_points[remap_merges,
                                    (1-close_merge)[remap_merges], :]
    # If no candidates are available, return an empty array
    if len(far_points) == 0:
        return np.zeros((0, 2))

    # this is the connected component label for the close_merge end of the merge point
    closest_label = cc_labels[np.arange(0, len(cc_labels)), close_merge]
    # use a query_ball_point to find all the potential 'close' partners to the
    # far point
    close_points = tree.query_ball_point(far_points, close_map_distance)

    # iterate over the far points
    for k, (close_cc, arr) in enumerate(zip(closest_label[remap_merges], close_points)):
        # k is the index of the remap candidate
        # close_cc is the connected component of the other side of that edge
        # arr is the list of candidate triangle center points

        # calculate a binary vector over the candidate points as to whether it is part
        # of a different connected component than the other side of the merge edge
        other_comp_points = labels[mesh.faces[arr, 0]] != close_cc
        # find out how far each of the candidates that is on a different connected component is
        d = np.linalg.norm(
            far_points[k] - mesh.triangles_center[arr][other_comp_points], axis=1)

        # if this is empty that there are no candidates and we can safely ignore this merge edge
        if (d.shape[0] != 0):
            # otherwise there is another connected component that would be a 'close' match
            # this reindexs k to the 'far' point on the edge into the linear index of 'close_inds'
            ind = remap_merges[k]*2 + (1-close_merge[remap_merges[k]])
            # remap the index of the closest point with a different connected component
            # into the mesh vertex index.
            close_inds[ind] = mesh.faces[arr[np.where(
                other_comp_points)[0][np.argmin(d)]], 0]
            # reset the is_join_merge index to True for this case
            is_join_merge[remap_merges[k]] = True

    # return the close_inds reshaped into Nmerge x 2, and filtered to only be those
    # that are connected differente connected components
    return close_inds.reshape((Nmerge, 2))[is_join_merge, :]


def merge_log_to_points(merge_log, base_resolution):
    """Process the raw merge log into points in the euclidean space.
    """
    if isinstance(merge_log, dict):
        ml = merge_log['merge_edge_coords']
    else:
        ml = merge_log
    merge_event_points = np.array(ml).squeeze()

    if len(merge_event_points.shape) != 3:
        merge_event_points = merge_event_points.reshape(-1, 2, 3)

    return merge_event_points * base_resolution


def merge_log_edges(mesh, merge_log, base_resolution, close_map_distance=300, verbose=False):
    """ Process a merge log into mesh link edges
    """
    merge_event_points = merge_log_to_points(merge_log, base_resolution)

    # map these merge edge coordinates to indices on the mesh
    if len(merge_event_points) > 0:
        merge_edge_inds = merge_points_to_merge_indices(mesh,
                                                        merge_event_points,
                                                        close_map_distance=close_map_distance)
    else:
        merge_edge_inds = []

    if verbose:
        print(len(merge_edge_inds), len(merge_event_points))
    # iterate over these merge indices and find the minimal edge
    # that links these two connected componets
    total_link_edges = []

    for merge_ind in merge_edge_inds:
        link_edges = find_edges_to_link(mesh, merge_ind[0], merge_ind[1])
        total_link_edges.append(link_edges)

    # reshape the combined result into a M x 2 matrix
    if len(total_link_edges) > 0:
        link_edges = np.concatenate(total_link_edges)
    else:
        link_edges = np.array([])
    link_edges = link_edges.reshape((len(link_edges), 2))

    return link_edges


def get_link_edges(mesh, seg_id, datastack_name=None, close_map_distance=300,
                   server_address=None, verbose=False, client=None):
    """function to get a set of edges that should be added to a mesh

    Parameters
    ----------
    mesh : trimesh_io.Mesh
        the mesh to add edges to
    seg_id : np.uint64 or int
        the seg_id to query the PCG endpoint for merges
    dataset_name: str
        the name of the dataset to query
    close_map_distance: int or float
        the maximum distance to map (default 300 in units of mesh.vertices)
    server_address: str
        the url to the root of the CAVE deployment
    verbose: bool
        whether to print debug statements
    client : caveclient.ChunkedGraphClient

    Returns
    -------
    np.array
        link_edges, a Kx2 array of mesh.vertices indices representing edges that should be added to the mesh graph

    """

    # initialize a chunkedgraph client
    if client is None:
        client = CAVEclient(
            datastack_name, server_address=server_address).chunkedgraph

    # get the merge log
    if type(seg_id) is np.int64:
        seg_id = int(seg_id)

    merge_log = client.get_merge_log(seg_id)

    return merge_log_edges(mesh, merge_log, client.base_resolution,
                           close_map_distance=close_map_distance,
                           verbose=verbose)
