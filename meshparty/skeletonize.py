from scipy import sparse
import numpy as np
import time


def get_path(root, target, pred):
    path = [target]
    p = target
    while p != root:
        p = pred[p]
        path.append(p)
    path.reverse()
    return path


def find_far_points(trimesh):
    d = 0
    dn = 1
    a = 0
    b = 1
    k = 0
    pred = None
    ds = None
    while 1:
        k += 1
        dsn, predn = sparse.csgraph.dijkstra(
            trimesh.csgraph, False, a, return_predecessors=True)
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


def setup_root(mesh, soma_pos=None, soma_thresh=7500):
    valid = np.ones(len(mesh.vertices), np.bool)

    # soma mode
    if soma_pos is not None:
        # pick the first soma as root
        soma_d, soma_i = mesh.kdtree.query(soma_pos,
                                           k=len(mesh.vertices),
                                           distance_upper_bound=soma_thresh)
        assert(np.min(soma_d)<soma_thresh)
        root = soma_i[np.argmin(soma_d)]
        valid[soma_i[~np.isinf(soma_d)]] = False
        root_ds, pred = sparse.csgraph.dijkstra(mesh.csgraph,
                                                False,
                                                root,
                                                return_predecessors=True)
    else:
        # there is no soma so use far point heuristic
        root, target, pred, dm, root_ds = find_far_points(mesh)
        valid[root] = 0

    return root, root_ds, pred, valid


def mesh_teasar(mesh, root=None, valid=None, root_ds=None, soma_pt=None,
                soma_thresh=7500, invalidation_d=10000, return_timing=False):
    # if no root passed, then calculation one
    if root is None:
        root, root_ds, pred, valid = setup_root(mesh,
                                                soma_pos=soma_pt,
                                                soma_thresh=soma_thresh)
    # if root_ds have not be precalculated do so
    if root_ds is None:
        root_ds, pred = sparse.csgraph.dijkstra(mesh.csgraph,
                                                False,
                                                root,
                                                return_predecessors=True)
    # if certain vertices haven't been pre-invalidated start with just
    # the root vertex invalidated
    if valid is None:
        valid = np.ones(len(mesh.vertices), np.bool)
        valid[root] = None
    else:
        if (len(valid) != len(mesh.vertices)):
            raise Exception("valid must be length of vertices")

    if not np.all(~np.isinf(root_ds)):
        raise Exception("all points should be reachable from root")

    # vector to store each branch result
    paths = []

    # vector to store each path's total length
    path_lengths = []

    # keep track of the nodes that have been visited
    visited_nodes = [root]

    # counter to track how many branches have been counted
    k = 0

    # arrays to track timing
    start = time.time()
    time_arrays = [[], [], [], [], [], []]

    # keep looping till all vertices have been invalidated
    while(np.sum(valid) > 0):
        k += 1

        t = time.time()
        # find the next target, farthest vertex from root
        # that has not been invalidated
        target = np.argmax(root_ds*valid)
        time_arrays[0].append(time.time()-t)

        t = time.time()
        # calculate the shortest path to that vertex
        # from all other vertices
        # up till the distance to the root
        ds, pred_t = sparse.csgraph.dijkstra(
            mesh.csgraph,
            False,
            target,
            limit=root_ds[target],
            return_predecessors=True)

        # pick out the vertex that has already been visited
        # which has the shortest path to target
        min_node = np.argmin(ds[visited_nodes])
        # reindex to get its absolute index
        branch = visited_nodes[min_node]
        # this is in the index of the point on the skeleton
        # we want this branch to connect to
        time_arrays[1].append(time.time()-t)

        t = time.time()
        # get the path from the target to branch point
        path = get_path(target, branch, pred_t)
        visited_nodes += path[0:-1]
        # record its length
        path_lengths.append(ds[branch])
        # record the path
        paths.append(path)
        time_arrays[2].append(time.time()-t)

        t = time.time()
        # get the distance to all points along the new path
        # within the invalidation distance
        dm = sparse.csgraph.dijkstra(
            mesh.csgraph, False, path, limit=invalidation_d)
        time_arrays[3].append(time.time()-t)

        t = time.time()
        # find the shortest distance to each mesh node
        dsp = np.min(dm, axis=0)
        time_arrays[4].append(time.time()-t)

        t = time.time()
        # all such non infinite distances are within the invalidation
        # zone and should be marked invalid
        valid[~np.isinf(dsp)] = False

        # print out how many vertices are still valid
        print(np.sum(valid))
        time_arrays[5].append(time.time()-t)
    # record the total time
    dt = time.time() - start

    if return_timing:
        return paths, path_lengths, time_arrays, dt
    else:
        return paths, path_lengths


def paths_to_edges(path_list):
    arrays = []
    for path in path_list:
        p = np.array(path)
        e = np.vstack((p[0:-1], p[1:])).T
        arrays.append(e)
    return np.vstack(arrays)


def smooth_graph(verts, edges, neighborhood=2, iterations=100, r=.1):
    """ smooths a spatial graph via iterative local averaging
        calculates the average position of neighboring vertices
        and relaxes the vertices toward that average

        :param verts: a NxK numpy array of vertex positions
        :param edges: a Mx2 numpy array of vertex indices that are edges
        :param neighborhood: an integer of how far in the graph to relax over
        as being local to any vertex (default = 2)
        :param iterations: number of relaxation iterations (default = 100)
        :param r: relaxation factor at each iteration
        new_vertex = (1-r)*old_vertex + r*(local_avg)

        :return: new_verts
        verts is a Nx3 list of new smoothed vertex positions

    """
    N = len(verts)
    E = len(edges)
    # setup a sparse matrix with the edges
    sm = sparse.csgraph.csc_matrix(
        (np.ones(E), (edges[:, 0], edges[:, 1])), shape=(N, N))

    # an identity matrix
    eye = sparse.csgraph.csc_matrix((np.ones(N, dtype=np.float32),
                                    (np.arange(0, N), np.arange(0, N))),
                                    shape=(N, N))
    # for undirected graphs we want it symettric
    sm = sm + sm.T

    # this will store our relaxation matrix
    C = sparse.csgraph.csc_matrix(eye)
    # multiple the matrix and add to itself
    # to spread connectivity along the graph
    for i in range(neighborhood):
        C = C + sm @ C
    # zero out the diagonal elements
    C.setdiag(np.zeros(N))
    # don't overweight things that are connected in more than one way
    C = C.sign()
    # measure total effective neighbors per node
    neighbors = np.sum(C, axis=1)

    # normalize the weights of neighbors according to number of neighbors
    neighbors = 1/neighbors
    C = C.multiply(neighbors)
    # convert back to csc
    C = C.tocsc()

    # multiply weights by relaxation term
    C *= r

    # construct relaxation matrix, adding on identity with complementary weight
    A = C + (1-r)*eye

    # make a copy of original vertices to no destroy inpuyt
    new_verts = np.copy(verts)

    # iteratively relax the vertices
    for i in range(iterations):
        new_verts = A*new_verts
    return new_verts
