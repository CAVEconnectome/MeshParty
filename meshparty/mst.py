import pcst_fast
import numpy as np
from scipy.sparse import csc_matrix


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


def smooth_graph(verts, edges, neighborhood = 2, iterations=100, r=.1):
    """ smooths a spatial graph via iterative local averaging
        calculates the average position of neighboring vertices
        and relaxes the vertices toward that average

        :param verts: a NxK numpy array of vertex positions
        :param edges: a Mx2 numpy array of vertex indices that are edges
        :param neighborhood: an integer of how many edges in the graph to consider
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
    sm = csc_matrix(
        (np.ones(E), (edges[:, 0], edges[:, 1])), shape=(N, N))
    
    # an identity matrix
    eye= csc_matrix((np.ones(N, dtype=np.float32), (np.arange(0, N), np.arange(0,N))), shape=(N,N))
    # for undirected graphs we want it symettric
    sm = sm + sm.T
    
    # this will store our relaxation matrix
    C = csc_matrix(eye)
    # multiple the matrix  and add to itself to spread connectivity along the graph
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
    C = C.multiply( neighbors)
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