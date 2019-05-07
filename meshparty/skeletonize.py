from scipy import sparse, spatial
import numpy as np
import time
from meshparty import trimesh_vtk, utils
import pandas as pd
from pykdtree.kdtree import KDTree
import pcst_fast
from tqdm import trange, tqdm
from meshparty.trimesh_io import Mesh
from trimesh.ray import ray_pyembree
from collections import defaultdict


def recenter_verts(verts, edges, centers):
    edge_df = pd.DataFrame()
    edge_df['start_edge']=np.array(edges[:,0], np.int64)
    edge_df['end_edge']=np.array(edges[:,1], np.int64)
    edge_df['center_x']=np.array(centers)[:,0]
    edge_df['center_y']=np.array(centers)[:,1]
    edge_df['center_z']=np.array(centers)[:,2]
    start_mean = edge_df.groupby('start_edge').mean()[['center_x','center_y','center_z']]
    new_verts = np.copy(verts)
    new_verts[start_mean.index.values,:]=start_mean.values
    return new_verts


def reduce_verts(verts, faces):
    """removes unused vertices from a graph or mesh

    verts = NxD numpy array of vertex locations
    faces = MxK numpy array of connected shapes (i.e. edges or tris)
    (entries are indices into verts)

    returns: 
    new_verts, new_face, used_verts
    a filtered set of vertices and reindexed set of faces
    along with the index of the new_verts in the old verst
    """
    used_verts = np.unique(faces.ravel())
    new_verts = verts[used_verts, :]
    new_face = np.zeros(faces.shape, dtype=faces.dtype)
    for i in range(faces.shape[1]):
        new_face[:, i] = np.searchsorted(used_verts, faces[:, i])
    return new_verts, new_face, used_verts

def skeletonize_mesh(mesh, soma_pt=None, soma_thresh=7500,
                invalidation_d=10000, smooth_neighborhood=5,
                max_tip_d=2000, large_skel_path_threshold=5000,
                cc_vertex_thresh=100, 
                collapse_soma=True,
                merge_components_at_tips=True, return_map=False):
    """ function to turn a trimesh object of a neuron into a skeleton
    Parameters
    ----------
    mesh: meshparty.trimesh_io.Mesh
        the mesh to skeletonize, defaults assume vertices in nm
    soma_pt: np.array
        a length 3 array specifying to soma location to make the root
        default=None, in which case a heuristic root will be chosen
        in units of mesh vertices
    soma_thresh: float
        distance in mesh vertex units over which to consider mesh 
        vertices close to soma_pt to belong to soma
        these vertices will automatically be invalidated and no
        skeleton branches will attempt to reach them.
        This distance will also be used to collapse all skeleton
        points within this distance to the soma_pt root if collpase_soma
        is true. (default=7500 (nm))
    invalidation_d: float
        the distance along the mesh to invalidate when applying TEASAR
        like algorithm.  Controls how detailed a structure the skeleton
        algorithm reaches. default (10000 (nm))
    smooth_neighborhood: int
        the neighborhood in edge hopes over which to smooth skeleton locations.
        This controls the smoothing of the skeleton
        (default 5)
    max_tip_d: float
        the maximum distance to consider merging tips over.
        This controls the MST based tip merging algorithm presently implemented
        (default 100 (nm))
    large_skel_path_threshold: int
        the threshold in terms of skeleton vertices that skeletons will be
        nominated for tip merging.  Smaller skeleton fragments 
        will not be merged at their tips (default 5000)
    cc_vertex_thresh: int
        the threshold in terms of vertex numbers that connected components
        of the mesh will be considered for skeletonization. mesh connected
        components with fewer than these number of vertices will be ignored
        by skeletonization algorithm. (default 100)
    collapse_soma: bool
        whether to collapse all skeleton vertices within soma_d of soma_pt
        to soma_pt. Only applies if soma_pt passed. (default True)
    merge_components_at_tips: bool
        whether to perform tip merging as a post processing step
        (default True)
    return_map: bool
        whether to return a map of how each mesh vertex maps onto each skeleton vertex
        based upon how it was invalidated.

    Returns
        skel_verts: np.array
            a Nx3 matrix of skeleton vertex positions
        skel_edges: np.array
            a Kx2 matrix of skeleton edge indices into skel_verts
        smooth_verts: np.array
            a Nx3 matrix of vertex positions after smoothing
        skel_verts_orig: np.array
            a N long index of skeleton vertices in the original mesh vertex index
        (mesh_to_skeleton_map): np.array
            a Mx2 map of mesh vertex indices to skeleton vertex indices
    """

    skeletonize_output = skeletonize_components(mesh,
                                                soma_pt=soma_pt,
                                                soma_thresh=soma_thresh,
                                                invalidation_d=invalidation_d,
                                                cc_vertex_thresh=cc_vertex_thresh,
                                                return_map=return_map)
    if return_map is True:
        all_paths, roots, tot_path_lengths, mesh_to_skeleton_map = skeletonize_output
    else:
        all_paths, roots, tot_path_lengths = skeletonize_output

    if merge_components_at_tips is True:
        tot_edges = merge_tips(mesh, all_paths, roots, tot_path_lengths,
                               large_skel_path_threshold=large_skel_path_threshold, max_tip_d=max_tip_d)
    else:
        all_edges = []
        for comp_paths in all_paths:
            all_edges.append(utils.paths_to_edges(comp_paths))
        tot_edges = np.vstack(all_edges)

    skel_verts, skel_edges, skel_verts_orig = reduce_verts(mesh.vertices, tot_edges)
    smooth_verts = smooth_graph(skel_verts, skel_edges, neighborhood=smooth_neighborhood)

    if return_map:
        mesh_to_skeleton_map = utils.nanfilter_shapes(np.unique(tot_edges.ravel()), mesh_to_skeleton_map)
    else:
        mesh_to_skeleton_map = None

    if collapse_soma:
        collapse_out = collapse_soma_skeleton(soma_pt, skel_verts, skel_edges,
                                              soma_d_thresh=soma_thresh,
                                              mesh_to_skeleton_map=mesh_to_skeleton_map)
        if mesh_to_skeleton_map is None:
            (skel_verts, skel_edges) = collapse_out
        else:
            (skel_verts, skel_edges, mesh_to_skeleton_map) = collapse_out
    
    output_tuple = (skel_verts, skel_edges, smooth_verts, skel_verts_orig)

    if return_map:
        output_tuple = output_tuple + (mesh_to_skeleton_map,)

    return output_tuple


def skeletonize(mesh_meta, seg_id, soma_pt=None, soma_thresh=7500,
                invalidation_d=10000, smooth_neighborhood=5,
                max_tip_d=2000, large_skel_path_threshold=5000,
                cc_vertex_thresh=100, do_cross_section=False,
                merge_components_at_tips=True, return_map=False):
    

    mesh = mesh_meta.mesh(seg_id=seg_id,
                          merge_large_components=False,
                          remove_duplicate_vertices=False)

    return skeletonize_mesh(mesh, soma_pt=soma_pt, soma_thresh=soma_thresh,
                invalidation_d=invalidation_d, smooth_neighborhood=smooth_neighborhood,
                max_tip_d=max_tip_d, large_skel_path_threshold=large_skel_path_threshold,
                cc_vertex_thresh=cc_vertex_thresh, do_cross_section=do_cross_section,
                merge_components_at_tips=merge_components_at_tips, return_map=return_map)


def skeletonize_axon(mesh_meta, axon_id, invalidation_d=5000, smooth_neighborhood=5,
                     max_tip_d=2000, large_skel_path_threshold=5000, cc_vertex_thresh=100,
                     return_map=False):
    return skeletonize(mesh_meta, axon_id,
                       invalidation_d=invalidation_d,
                       smooth_neighborhood=smooth_neighborhood,
                       max_tip_d=max_tip_d,
                       large_skel_path_threshold=large_skel_path_threshold,
                       cc_vertex_thresh=cc_vertex_thresh,
                       return_map=return_map)


def merge_tips(mesh, all_paths, roots, tot_path_lengths,
               large_skel_path_threshold=5000, max_tip_d=2000):

    # collect all the tips of the skeletons (including roots)
    skel_tips = []
    all_tip_indices = []
    for paths, root in zip(all_paths, roots):
        tips = []
        tip_indices = []
        for path in paths:
            tip_ind = path[0]
            tip = mesh.vertices[tip_ind, :]
            tips.append(tip)
            tip_indices.append(tip_ind)
        root_tip = mesh.vertices[root, :]
        tips.append(root_tip)
        tip_indices.append(root)
        skel_tips.append(np.vstack(tips))
        all_tip_indices.append(np.array(tip_indices))
    # this is our overall tip matrix merged together
    all_tips = np.vstack(skel_tips)
    # and the vertex index of those tips in the original mesh
    all_tip_indices = np.concatenate(all_tip_indices)
    
    # variable to keep track of what component each tip was from
    tip_component = np.zeros(all_tips.shape[0])
    # counter to keep track of an overall tip index as we go through 
    # the components with different numbers of tips
    ind_counter = 0

    # setup the prize collection steiner forest problem variables
    # prizes will be related to path length of the tip components
    tip_prizes = [] 
    # where to collect all the tip<>tip edges
    all_edges = []
    # where to collect all the tip<>tip edge weights
    all_edge_weights = []

    # loop over all the components and their tips
    for k, tips, path_lengths in zip(range(len(tot_path_lengths)), skel_tips, tot_path_lengths):
        # how many tips in this component
        ntips = tips.shape[0]
        # calculate the total path length in this component
        path_len = np.sum(np.array(path_lengths))
        # the prize is 0 if this is small, and the path length if big
        prize = path_len if path_len > large_skel_path_threshold else 0
        # the cost of traveling within a skeleton is 0 if big, and the path_len if small
        cost = path_len if path_len <= large_skel_path_threshold else 0
        # add a block of prizes to the tip prizes for this component
        tip_prizes.append(prize*np.ones(ntips))
        # make an array of overall tip index for this component
        comp_tips = np.arange(ind_counter, ind_counter+ntips, dtype=np.int64)
        # add edges between this components root and each of the tips
        root_tips = (ind_counter+ntips-1)*np.ones(ntips, dtype=np.int64)
        in_tip_edges = np.hstack([root_tips[:, np.newaxis],
                                  comp_tips[:, np.newaxis]])
        all_edges.append(in_tip_edges)

        # add a block for the cost of these edges
        all_edge_weights.append(cost*np.ones(ntips))
        # note what component each of these tips is from
        tip_component[comp_tips] = k
        # increment our overall index counter
        ind_counter += ntips
    # gather all the prizes into a single block
    tip_prizes = np.concatenate(tip_prizes)

    # make a kdtree with all the tips
    tip_tree = spatial.cKDTree(all_tips)
    
    # find the tips near one another
    close_tips = tip_tree.query_pairs(max_tip_d, output_type='ndarray')
    # filter out close tips from the same component
    diff_comp = ~(tip_component[close_tips[:, 0]] == tip_component[close_tips[:, 1]])
    filt_close_tips = close_tips[diff_comp]

    # add these as edges
    all_edges.append(filt_close_tips)
    # with weights equal to their euclidean distance
    dv = np.linalg.norm(all_tips[filt_close_tips[:,0],:] - all_tips[filt_close_tips[:, 1]], axis=1)
    all_edge_weights.append(dv)

    # consolidate the edges and weights into a single array
    inter_tip_weights = np.concatenate(all_edge_weights)
    inter_tip_edges = np.concatenate(all_edges)

    # run the prize collecting steiner forest optimization
    mst_verts, mst_edges = pcst_fast.pcst_fast(
        inter_tip_edges, tip_prizes, inter_tip_weights, -1, 1, 'gw', 1)
#     # find the set of mst edges that are between connected components
    new_mst_edges = mst_edges[tip_component[inter_tip_edges[mst_edges, 0]] != tip_component[inter_tip_edges[mst_edges ,1]]]
    good_inter_tip_edges = inter_tip_edges[new_mst_edges, :]
    # get these in the original index
    new_edges_orig_ind = all_tip_indices[good_inter_tip_edges]
#     # collect all the edges for all the paths into a single list
#     # with the original indices of the mesh
    orig_edges = []
    for paths, root in zip(all_paths, roots):
        edges = utils.paths_to_edges(paths)
        orig_edges.append(edges)
    orig_edges = np.vstack(orig_edges)
    # and add our new mst edges
    tot_edges = np.vstack([orig_edges, new_edges_orig_ind])

    return tot_edges


# def fix_skeleton(verts, edges):
#     # fix the skeleton so that it    
#     # filter out the vertices to be just those included in this skeleton
#     skel_verts, skel_edges = trimesh_vtk.remove_unused_verts(axon_trimesh.vertices, tot_edges)
#     Nind = skel_verts.shape[0]
#     g=sparse.csc_matrix((np.ones(len(skel_edges)), (skel_edges[:,0], skel_edges[:,1])), shape =
#                         (Nind,Nind))

#     # figure out how many skeleton components we have now
#     n_skel_comp, skel_labels = sparse.csgraph.connected_components(g,directed=False, return_labels=True)
#     skel_comp_labels, skel_comp_counts = np.unique(skel_labels, return_counts = True)
#     larg_skel_cc_ind = np.where(skel_comp_counts>100)[0]
#     large_skel_components=len(larg_skel_cc_ind)


def skeletonize_components(mesh,
                           soma_pt=None,
                           soma_thresh=10000,
                           invalidation_d=10000,
                           cc_vertex_thresh=100,
                           return_map=False):
    # find all the connected components in the mesh
    n_components, labels = sparse.csgraph.connected_components(mesh.csgraph,
                                                               directed=False,
                                                               return_labels=True)
    comp_labels, comp_counts = np.unique(labels, return_counts=True)

    if return_map:
        mesh_to_skeleton_map = np.full(len(mesh.vertices), np.nan)

    # variables to collect the paths, roots and path lengths
    all_paths = []
    roots = []
    tot_path_lengths = []

    if soma_pt is not None:
        soma_d = mesh.vertices - soma_pt[np.newaxis, :]
        soma_d = np.linalg.norm(soma_d, axis=1)
        is_soma_pt = soma_d < soma_thresh
    else:
        is_soma_pt = None
        soma_d = None
    # loop over the components
    for k in trange(n_components):
        if comp_counts[k] > cc_vertex_thresh:

            # find the root using a soma position if you have it
            # it will fall back to a heuristic if the soma
            # is too far away for this component
            root, root_ds, pred, valid = setup_root_new(mesh,
                                                        is_soma_pt,
                                                        soma_d,
                                                        labels == k)
            # run teasar on this component
            teasar_output = mesh_teasar(mesh,
                                        root=root,
                                        root_ds=root_ds,
                                        root_pred=pred,
                                        valid=valid,
                                        invalidation_d=invalidation_d,
                                        return_map=return_map)
            if return_map is False:
                paths, path_lengths = teasar_output
            else:
                paths, path_lengths, mesh_to_skeleton_map_single = teasar_output
                mesh_to_skeleton_map[~np.isnan(mesh_to_skeleton_map_single)] = mesh_to_skeleton_map_single[~np.isnan(mesh_to_skeleton_map_single)]

  
            if len(path_lengths) > 0:
                # collect the results in lists
                tot_path_lengths.append(path_lengths)
                all_paths.append(paths)
                roots.append(root)
            
    if return_map:
        return all_paths, roots, tot_path_lengths, mesh_to_skeleton_map
    else:
        return all_paths, roots, tot_path_lengths


def setup_root_new(mesh, is_soma_pt=None, soma_d=None, is_valid=None):
    if is_valid is not None:
        valid = np.copy(is_valid)
    else:
        valid = np.ones(len(mesh.vertices), np.bool)
    assert(len(valid)==mesh.vertices.shape[0])

    root = None
    # soma mode
    if is_soma_pt is not None:
        # pick the first soma as root
        assert(len(soma_d)==mesh.vertices.shape[0])
        assert(len(is_soma_pt)==mesh.vertices.shape[0])
        is_valid_root = is_soma_pt & valid
        valid_root_inds = np.where(is_valid_root)[0]
        if len(valid_root_inds) > 0:
            min_valid_root = np.nanargmin(soma_d[valid_root_inds])
            root = valid_root_inds[min_valid_root]           
            root_ds, pred = sparse.csgraph.dijkstra(mesh.csgraph,
                                                    directed=False,
                                                    indices=root,
                                                    return_predecessors=True)
        else:
            start_ind = np.where(valid)[0][0]
            root, target, pred, dm, root_ds = utils.find_far_points(mesh,
                                                                    start_ind=start_ind)
        valid[is_soma_pt] = False

    if root is None:
        # there is no soma close, so use far point heuristic
        start_ind = np.where(valid)[0][0]
        root, target, pred, dm, root_ds = utils.find_far_points(mesh, start_ind=start_ind)
    valid[root] = False
    assert(np.all(~np.isinf(root_ds[valid])))
    return root, root_ds, pred, valid


def setup_root(mesh, soma_pt=None, soma_thresh=7500, valid_inds=None):
    if valid_inds is not None:
        valid = np.zeros(len(mesh.vertices), np.bool)
        valid[valid_inds] = True
    else:
        valid = np.ones(len(mesh.vertices), np.bool)
    root = None
    # soma mode
    if soma_pt is not None:
        # pick the first soma as root
        soma_d, soma_i = mesh.kdtree.query(soma_pt,
                                           k=len(mesh.vertices),
                                           distance_upper_bound=soma_thresh)
        if valid_inds is not None:
            soma_i, valid_soma_ind, soma_valid_ind = np.intersect1d(soma_i,
                                                                    valid_inds,
                                                                    return_indices=True)
            soma_d = soma_d[valid_soma_ind]
        if (len(soma_d) > 0):
            min_d = np.min(soma_d)
        else:
            min_d = np.inf 
        if (min_d < soma_thresh):
            root = soma_i[np.argmin(soma_d)]
            root_ds, pred = sparse.csgraph.dijkstra(mesh.csgraph,
                                                    False,
                                                    root,
                                                    return_predecessors=True)
        else:
            if valid_inds is not None:
                root, target, pred, dm, root_ds = utils.find_far_points(mesh, start_ind=valid_inds[0])
            else:
                root, target, pred, dm, root_ds = utils.find_far_points(mesh)
    if root is None:
        # there is no soma close, so use far point heuristic
        root, target, pred, dm, root_ds = utils.find_far_points(mesh)
    valid[root] = False

    return root, root_ds, pred, valid


def mesh_teasar(mesh, root=None, valid=None, root_ds=None, root_pred=None, soma_pt=None,
                soma_thresh=7500, invalidation_d=10000, return_timing=False, return_map=False):
    # if no root passed, then calculation one
    if root is None:
        root, root_ds, root_pred, valid = setup_root(mesh,
                                                     soma_pt=soma_pt,
                                                     soma_thresh=soma_thresh)
    # if root_ds have not be precalculated do so
    if root_ds is None:
        root_ds, root_pred = sparse.csgraph.dijkstra(mesh.csgraph,
                                                     False,
                                                     root,
                                                     return_predecessors=True)
    # if certain vertices haven't been pre-invalidated start with just
    # the root vertex invalidated
    if valid is None:
        valid = np.ones(len(mesh.vertices), np.bool)
        valid[root] = False
    else:
        if (len(valid) != len(mesh.vertices)):
            raise Exception("valid must be length of vertices")

    if return_map == True:
        mesh_to_skeleton_dist = np.full(len(mesh.vertices), np.inf)
        mesh_to_skeleton_map = np.full(len(mesh.vertices), np.nan)

    total_to_visit = np.sum(valid)
    if np.sum(np.isinf(root_ds) & valid) != 0:
        print(np.where(np.isinf(root_ds) & valid))
        raise Exception("all valid vertices should be reachable from root")

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
    time_arrays = [[], [], [], [], []]

    with tqdm(total=total_to_visit) as pbar:
        # keep looping till all vertices have been invalidated
        while(np.sum(valid) > 0):
            k += 1
            t = time.time()
            # find the next target, farthest vertex from root
            # that has not been invalidated
            target = np.nanargmax(root_ds*valid)
            if (np.isinf(root_ds[target])):
                raise Exception('target cannot be reached')
            time_arrays[0].append(time.time()-t)

            t = time.time()
            # figure out the longest this branch could be
            # by following the route from target to the root
            # and finding the first already visited node (max_branch)
            # The dist(root->target) - dist(root->max_branch)
            # is the maximum distance the shortest route to a branch
            # point from the target could possibly be,
            # use this bound to reduce the djisktra search radius for this target
            max_branch = target
            while max_branch not in visited_nodes:
                max_branch = root_pred[max_branch]
            max_path_length = root_ds[target]-root_ds[max_branch]

            # calculate the shortest path to that vertex
            # from all other vertices
            # up till the distance to the root
            ds, pred_t = sparse.csgraph.dijkstra(
                mesh.csgraph,
                False,
                target,
                limit=max_path_length,
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
            path = utils.get_path(target, branch, pred_t)
            visited_nodes += path[0:-1]
            # record its length
            assert(~np.isinf(ds[branch]))
            path_lengths.append(ds[branch])
            # record the path
            paths.append(path)
            time_arrays[2].append(time.time()-t)

            t = time.time()
            # get the distance to all points along the new path
            # within the invalidation distance
            dm, _, sources = sparse.csgraph.dijkstra(
                mesh.csgraph, False, path, limit=invalidation_d,
                min_only=True, return_predecessors=True)
            time_arrays[3].append(time.time()-t)

            t = time.time()
            # all such non infinite distances are within the invalidation
            # zone and should be marked invalid
            nodes_to_update = ~np.isinf(dm)
            marked = np.sum(valid & ~np.isinf(dm))
            if return_map == True:
                new_sources_closer = dm[nodes_to_update]<mesh_to_skeleton_dist[nodes_to_update]
                mesh_to_skeleton_map[nodes_to_update] = np.where(new_sources_closer,
                                                                 sources[nodes_to_update],
                                                                 mesh_to_skeleton_map[nodes_to_update])
                mesh_to_skeleton_dist[nodes_to_update] = np.where(new_sources_closer,
                                                                  dm[nodes_to_update],
                                                                  mesh_to_skeleton_dist[nodes_to_update])

            valid[~np.isinf(dm)] = False


            # print out how many vertices are still valid
            pbar.update(marked)
            time_arrays[4].append(time.time()-t)
    # record the total time
    dt = time.time() - start

    out_tuple = (paths, path_lengths)
    if return_map:
        out_tuple = out_tuple + (mesh_to_skeleton_map,)
    if return_timing:
        out_tuple = out_tuple + (time_arrays, dt)

    return out_tuple


def smooth_graph(verts, edges, mask=None, neighborhood=2, iterations=100, r=.1):
    """ smooths a spatial graph via iterative local averaging
        calculates the average position of neighboring vertices
        and relaxes the vertices toward that average

        :param verts: a NxK numpy array of vertex positions
        :param edges: a Mx2 numpy array of vertex indices that are edges
        :param mask: optional N boolean vector of values to mask
        the vert locations.  the result will return a result at every vert
        but the values that are false in this mask will be ignored and not
        factored into the smoothing.
        :param neighborhood: an integer of how far in the graph to relax over
        as being local to any vertex (default = 2)
        :param iterations: number of relaxation iterations (default = 100)
        :param r: relaxation factor at each iteration
        new_vertex = (1-r)*old_vertex*mask + (r+(1-r)*(1-mask))*(local_avg)

        :return: new_verts
        verts is a Nx3 list of new smoothed vertex positions

    """
    N = len(verts)
    E = len(edges)

    # setup a sparse matrix with the edges
    sm = sparse.csc_matrix(
        (np.ones(E), (edges[:, 0], edges[:, 1])), shape=(N, N))

    # an identity matrix
    eye = sparse.csc_matrix((np.ones(N, dtype=np.float32),
                                    (np.arange(0, N), np.arange(0, N))),
                                    shape=(N, N))
    # for undirected graphs we want it symettric
    sm = sm + sm.T

    # this will store our relaxation matrix
    C = sparse.csc_matrix(eye)
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


def collapse_soma_skeleton(soma_pt, verts, edges, soma_d_thresh=12000, mesh_to_skeleton_map=None, soma_mesh_indices=None, return_filter=False, return_soma_ind=False):
    if soma_pt is not None:
        soma_pt_m = soma_pt[np.newaxis, :]
        dv = np.linalg.norm(verts - soma_pt_m, axis=1)
        soma_verts = np.where(dv < soma_d_thresh)[0]
        new_verts = np.vstack((verts, soma_pt_m))
        soma_i = verts.shape[0]
        edges_m = edges.copy()
        edges_m[np.isin(edges, soma_verts)] = soma_i
        
        simple_verts, simple_edges = trimesh_vtk.remove_unused_verts(new_verts, edges_m)
        good_edges = ~(simple_edges[:, 0] == simple_edges[:, 1])

        if mesh_to_skeleton_map is not None:
            new_mesh_to_skeleton_map = mesh_to_skeleton_map.copy()
            remap_rows = np.isin(mesh_to_skeleton_map, soma_verts)
            new_mesh_to_skeleton_map[remap_rows] = soma_i
            new_mesh_to_skeleton_map = utils.nanfilter_shapes(np.unique(edges_m.ravel()),
                                                                        new_mesh_to_skeleton_map)
            if soma_mesh_indices is not None:
                new_mesh_to_skeleton_map[soma_mesh_indices] = len(simple_verts)-1

        output = [simple_verts, simple_edges[good_edges]]
        if mesh_to_skeleton_map is not None:
            output.append(new_mesh_to_skeleton_map)
        if return_filter:
            used_vertices = np.unique(edges_m.ravel())[:-1]   #Remove the largest value which is soma_i
            output.append(used_vertices)
        if return_soma_ind:
            output.append(len(simple_verts)-1)
        return output

    else:
        simple_verts, simple_edges = trimesh_vtk.remove_unused_verts(verts, edges)
        return simple_verts, simple_edges

def skeleton_index_to_mesh_index_map(skel_map):
    '''
    Makes a dict indexed by skeleton vertex with mesh vertex indices that map to it.
    '''
    skind_to_mesh_map = defaultdict(list)
    sk_values = np.unique(skel_map[~np.isnan(skel_map)])
    nan_map_inds = np.isnan(skel_map)
    skel_map_nonan = skel_map[~nan_map_inds]
    map_inds_nonan = np.arange(0,len(skel_map))[~nan_map_inds]
    for ii, skind in zip(map_inds_nonan, skel_map_nonan):
        skind_to_mesh_map[skind].append(ii)
    return skind_to_mesh_map

def ray_trace_distance(vertex_inds, mesh, max_iter=10, rand_jitter=0.001, verbose=False, ray_inter=None):
    '''
    Compute distance to opposite side of the mesh for specified vertex indices on the mesh.
    '''
    if ray_inter is None:
        ray_inter = ray_pyembree.RayMeshIntersector(mesh)

    rs = np.zeros(len(vertex_inds))
    good_rs = np.full(len(rs), False)

    it= 0
    while not np.all(good_rs):
        if verbose:
            print(np.sum(~good_rs))
        blank_inds = np.where(~good_rs)[0]
        starts = (mesh.vertices-mesh.vertex_normals)[vertex_inds,:][~good_rs,:]
        vs = -mesh.vertex_normals[vertex_inds,:] \
              + (1.2**it)*rand_jitter*np.random.rand(*mesh.vertex_normals[vertex_inds,:].shape)                                                         
        vs = vs[~good_rs,:]

        rtrace = ray_inter.intersects_location(starts, vs, multiple_hits=False)
        
        if len(rtrace[0]>0):
            # radius values
            rs[blank_inds[rtrace[1]]] = np.linalg.norm(mesh.vertices[vertex_inds,:][rtrace[1]]-rtrace[0], axis=1)
            good_rs[blank_inds[rtrace[1]]]=True
        it+=1
        if it>max_iter:
            break
    return rs
