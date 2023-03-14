from scipy import sparse, spatial, optimize, signal
import numpy as np
import time
from meshparty import utils

try:
    from pykdtree.kdtree import KDTree
except:
    KDTree = spatial.cKDTree
from tqdm import tqdm
from meshparty.skeleton import Skeleton

import fastremap
import logging
from . import skeleton_utils


def skeletonize_mesh(
    mesh,
    soma_pt=None,
    soma_radius=7500,
    collapse_soma=True,
    collapse_function="sphere",
    invalidation_d=12000,
    smooth_vertices=False,
    compute_radius=True,
    shape_function="single",
    compute_original_index=True,
    verbose=True,
    smooth_iterations=12,
    smooth_neighborhood=2,
    smooth_r=0.1,
    cc_vertex_thresh=100,
    root_index=None,
    remove_zero_length_edges=True,
    collapse_params={},
    meta={},
):
    """
    Build skeleton object from mesh skeletonization

    Parameters
    ----------
    mesh: meshparty.trimesh_io.Mesh
        the mesh to skeletonize, defaults assume vertices in nm
    soma_pt: np.array
        a length 3 array specifying to soma location to make the root
        default=None, in which case a heuristic root will be chosen
        in units of mesh vertices.
    soma_radius: float
        distance in mesh vertex units over which to consider mesh
        vertices close to soma_pt to belong to soma
        these vertices will automatically be invalidated and no
        skeleton branches will attempt to reach them.
        This distance will also be used to collapse all skeleton
        points within this distance to the soma_pt root if collpase_soma
        is true. (default=7500 (nm))
    collapse_soma: bool
        whether to collapse the skeleton around the soma point (default True)
    collapse_function: 'sphere' or 'branch'
        Determines which soma collapse function to use. Sphere uses the soma_radius
        and collapses all vertices within that radius to the soma. Branch is an experimental
        approach that tries to compute the right boundary for each branch into soma.
    invalidation_d: float
        the distance along the mesh to invalidate when applying TEASAR
        like algorithm.  Controls how detailed a structure the skeleton
        algorithm reaches. default (12000 (nm))
    smooth_vertices: bool
        whether to smooth the vertices of the skeleton
    compute_radius: bool
        whether to calculate the radius of the skeleton at each point on the skeleton
        (default True)
    shape_function: 'single' or 'cone'
        Selects how to compute the radius, either with a single ray or a cone of rays. Default is 'single'.
    compute_original_index: bool
        whether to calculate how each of the mesh nodes maps onto the skeleton
        (default True)
    smooth_iterations: int, optional
        Number of iterations to smooth (default is 12)
    smooth_neighborhood: int, optional
        Size of neighborhood to look at for smoothing
    smooth_r: float, optional
        Weight of update step in smoothing algorithm, default is 0.2
    root_index: int or None, optional
        A precise mesh vertex to use as the skeleton root. If provided, the vertex location overrides soma_pt. By default, None.
    remove_zero_length_edges: bool
        If True, removes vertices involved in zero length edges, which can disrupt graph computations. Default True.
    collapse_params: dict
        Extra keyword arguments for the collapse function. See soma_via_sphere and soma_via_branch_starts for specifics.
    cc_vertex_thresh : int, optional
        Smallest number of vertices in a connected component to skeletonize.
    verbose: bool
        whether to print verbose logging
    meta: dict
        Skeletonization metadata to add to the skeleton. See skeleton.SkeletonMetadata for keys.

    Returns
    -------
    :obj:`meshparty.skeleton.Skeleton`
           a Skeleton object for this mesh
    """
    (
        skel_verts,
        skel_edges,
        orig_skel_index,
        skel_map,
    ) = calculate_skeleton_paths_on_mesh(
        mesh,
        invalidation_d=invalidation_d,
        cc_vertex_thresh=cc_vertex_thresh,
        root_index=root_index,
        return_map=True,
    )

    if smooth_vertices is True:
        smooth_verts = smooth_graph(
            skel_verts,
            skel_edges,
            neighborhood=smooth_neighborhood,
            iterations=smooth_iterations,
            r=smooth_r,
        )
        skel_verts = smooth_verts

    if root_index is not None and soma_pt is None:
        soma_pt = mesh.vertices[root_index]

    if soma_pt is not None:
        soma_pt = np.array(soma_pt).reshape(1, 3)

    rs = None

    if collapse_soma is True and soma_pt is not None:
        temp_sk = Skeleton(
            skel_verts,
            skel_edges,
            mesh_index=mesh.map_indices_to_unmasked(orig_skel_index),
            mesh_to_skel_map=skel_map,
        )
        _, close_ind = temp_sk.kdtree.query(soma_pt)
        temp_sk.reroot(close_ind[0])

        if collapse_function == "sphere":
            soma_verts, soma_r = soma_via_sphere(
                soma_pt, temp_sk.vertices, temp_sk.edges, soma_radius
            )
        elif collapse_function == "branch":
            try:
                from .ray_tracing import ray_trace_distance, shape_diameter_function
            except:
                raise ImportError('Could not import pyembree for ray tracing')

            if shape_function == "single":
                rs = ray_trace_distance(
                    mesh.filter_unmasked_indices_padded(temp_sk.mesh_index), mesh
                )
            elif shape_function == "cone":
                rs = shape_diameter_function(
                    mesh.filter_unmasked_indices_padded(temp_sk.mesh_index),
                    mesh,
                    num_points=30,
                    cone_angle=np.pi / 3,
                )

            soma_verts, soma_r = soma_via_branch_starts(
                temp_sk,
                mesh,
                soma_pt,
                rs,
                search_radius=collapse_params.get("search_radius", 25000),
                fallback_radius=collapse_params.get("fallback_radius", soma_radius),
                cutoff_threshold=collapse_params.get("cutoff_threshold", 0.4),
                min_cutoff=collapse_params.get("min_cutoff", 0.1),
                dynamic_range=collapse_params.get("dynamic_range", 1),
                dynamic_threshold=collapse_params.get("dynamic_threshold", False),
            )
        if root_index is not None:
            collapse_index = np.flatnonzero(orig_skel_index == root_index)[0]
        else:
            collapse_index = None
        new_v, new_e, new_skel_map, vert_filter, root_ind = collapse_soma_skeleton(
            soma_verts,
            soma_pt,
            temp_sk.vertices,
            temp_sk.edges,
            mesh_to_skeleton_map=temp_sk.mesh_to_skel_map,
            collapse_index=collapse_index,
            return_filter=True,
            return_soma_ind=True,
        )
    else:
        new_v, new_e, new_skel_map = skel_verts, skel_edges, skel_map
        vert_filter = np.arange(len(orig_skel_index))
        if root_index is not None:
            root_ind = np.flatnonzero(orig_skel_index == root_index)[0]
        elif soma_pt is None:
            sk_graph = utils.create_csgraph(new_v, new_e)
            root_ind = utils.find_far_points_graph(sk_graph)[0]
        else:
            # Still try to root close to the soma
            _, qry_inds = spatial.cKDTree(new_v, balanced_tree=False).query(
                soma_pt[np.newaxis, :]
            )
            root_ind = qry_inds[0]

    skel_map_full_mesh = np.full(mesh.node_mask.shape, -1, dtype=np.int64)
    skel_map_full_mesh[mesh.node_mask] = new_skel_map
    ind_to_fix = mesh.map_boolean_to_unmasked(np.isnan(new_skel_map))
    skel_map_full_mesh[ind_to_fix] = -1

    props = {}

    if compute_original_index is True:
        if collapse_soma is True and soma_pt is not None:
            mesh_index = temp_sk.mesh_index[vert_filter]
            if root_index is None:
                mesh_index = np.append(mesh_index, -1)
        else:
            mesh_index = orig_skel_index[vert_filter]
        props["mesh_index"] = mesh_index

    if compute_radius is True:
        if rs is None:
            try:
                from .ray_tracing import ray_trace_distance, shape_diameter_function
            except:
                raise ImportError('Could not import pyembree for ray tracing')

            if shape_function == "single":
                rs = ray_trace_distance(orig_skel_index[vert_filter], mesh)
            elif shape_function == "cone":
                rs = shape_diameter_function(orig_skel_index[vert_filter], mesh)
        else:
            rs = rs[vert_filter]
        if collapse_soma is True and soma_pt is not None:
            if root_index is None:
                rs = np.append(rs, soma_r)
            else:
                rs[root_ind] = soma_r
        props["rs"] = rs

    sk_params = {
        "soma_pt_x": soma_pt[0, 0] if soma_pt is not None else None,
        "soma_pt_y": soma_pt[0, 1] if soma_pt is not None else None,
        "soma_pt_z": soma_pt[0, 2] if soma_pt is not None else None,
        "soma_radius": soma_radius,
        "collapse_soma": collapse_soma,
        "collapse_function": collapse_function,
        "invalidation_d": invalidation_d,
        "smooth_vertices": smooth_vertices,
        "compute_radius": compute_radius,
        "shape_function": shape_function,
        "smooth_iterations": smooth_iterations,
        "smooth_neighborhood": smooth_neighborhood,
        "smooth_r": smooth_r,
        "cc_vertex_thresh": cc_vertex_thresh,
        "remove_zero_length_edges": remove_zero_length_edges,
        "collapse_params": collapse_params,
        "timestamp": time.time(),
    }
    sk_params.update(meta)

    sk = Skeleton(
        new_v,
        new_e,
        mesh_to_skel_map=skel_map_full_mesh,
        mesh_index=props.get("mesh_index", None),
        radius=props.get("rs", None),
        root=root_ind,
        remove_zero_length_edges=remove_zero_length_edges,
        meta=sk_params,
    )

    if compute_radius is True:
        _remove_nan_radius(sk)

    return sk


def _remove_nan_radius(sk, set_unfixed_to_lowest=True):

    last_numnans = np.inf
    nanlocs = np.flatnonzero(np.isnan(sk.radius))
    numnans = len(nanlocs)
    while numnans > 0 and last_numnans > numnans:
        for nanloc in nanlocs:
            sparse_row = sk.csgraph_binary_undirected[nanloc].toarray().ravel()
            prod = sparse_row * sk.radius
            with np.errstate(divide="ignore", invalid="ignore"):
                new_rad = np.nansum(prod) / np.nansum(prod > 0)
            sk._rooted.radius[nanloc] = new_rad
        last_numnans = numnans
        nanlocs = np.flatnonzero(np.isnan(sk.radius))
        numnans = len(nanlocs)

    if numnans > 0 and set_unfixed_to_lowest:
        sk._rooted.radius[nanlocs] = np.nanmin(sk.radius)


def calculate_skeleton_paths_on_mesh(
    mesh,
    soma_pt=None,
    soma_thresh=7500,
    invalidation_d=10000,
    smooth_neighborhood=2,
    smooth_iterations=12,
    cc_vertex_thresh=100,
    large_skel_path_threshold=5000,
    return_map=False,
    root_index=None,
):
    """function to turn a trimesh object of a neuron into a skeleton, without running soma collapse,
    or recasting result into a Skeleton.  Used by :func:`meshparty.skeletonize.skeletonize_mesh` and
    makes use of :func:`meshparty.skeletonize.skeletonize_components`

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
    large_skel_path_threshold: int
        the threshold in terms of skeleton vertices that skeletons will be
        nominated for tip merging.  Smaller skeleton fragments
        will not be merged at their tips (default 5000)
    cc_vertex_thresh: int
        the threshold in terms of vertex numbers that connected components
        of the mesh will be considered for skeletonization. mesh connected
        components with fewer than these number of vertices will be ignored
        by skeletonization algorithm. (default 100)
    return_map: bool
        whether to return a map of how each mesh vertex maps onto each skeleton vertex
        based upon how it was invalidated.
    root_index: int or None
        Mesh vertex to set as initial root node. Overides soma_pt if provided. Default is None.

    Returns
    -------
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

    skeletonize_output = skeletonize_components(
        mesh,
        soma_pt=soma_pt,
        soma_thresh=soma_thresh,
        invalidation_d=invalidation_d,
        cc_vertex_thresh=cc_vertex_thresh,
        return_map=return_map,
        root_index=root_index,
    )
    if return_map is True:
        all_paths, roots, tot_path_lengths, mesh_to_skeleton_map = skeletonize_output
    else:
        all_paths, roots, tot_path_lengths = skeletonize_output

    all_edges = []
    for comp_paths in all_paths:
        all_edges.append(utils.paths_to_edges(comp_paths))
    if len(all_edges) > 0:
        tot_edges = np.vstack(all_edges)
    else:
        tot_edges = np.zeros((3, 0))

    skel_verts, skel_edges, skel_verts_orig = reduce_verts(mesh.vertices, tot_edges)

    if return_map:
        mesh_to_skeleton_map = utils.nanfilter_shapes(
            np.unique(tot_edges.ravel()), mesh_to_skeleton_map
        )
        mesh_to_skeleton_map[np.isnan(mesh_to_skeleton_map)] = -1
    else:
        mesh_to_skeleton_map = None

    output_tuple = (skel_verts, skel_edges, skel_verts_orig)

    if return_map:
        output_tuple = output_tuple + (mesh_to_skeleton_map.astype(int),)

    return output_tuple


def reduce_verts(verts, faces):
    """removes unused vertices from a graph or mesh

    Parameters
    ----------
    verts : numpy.array
        NxD numpy array of vertex locations
    faces : numpy.array
        MxK numpy array of connected shapes (i.e. edges or tris)
        (entries are indices into verts)

    Returns
    -------
    np.array
        new_verts, a filtered set of vertices
    np.array
        new_face, a reindexed set of faces (or edges)
    np.array
        used_verts, the index of the new_verts in the old verts

    """
    used_verts = np.unique(faces.ravel())
    new_verts = verts[used_verts, :]
    new_face = np.zeros(faces.shape, dtype=faces.dtype)
    for i in range(faces.shape[1]):
        new_face[:, i] = np.searchsorted(used_verts, faces[:, i])
    return new_verts, new_face, used_verts


def skeletonize_components(
    mesh,
    soma_pt=None,
    soma_thresh=10000,
    invalidation_d=10000,
    cc_vertex_thresh=100,
    return_map=False,
    root_index=None,
):
    """core skeletonization routine, used by :func:`meshparty.skeletonize.calculate_skeleton_paths_on_mesh`
    to calculate skeleton on all components of mesh, with no post processing"""
    # find all the connected components in the mesh
    n_components, labels = sparse.csgraph.connected_components(
        mesh.csgraph, directed=False, return_labels=True
    )
    comp_labels, comp_counts = np.unique(labels, return_counts=True)

    if return_map:
        mesh_to_skeleton_map = np.full(len(mesh.vertices), np.nan)

    # variables to collect the paths, roots and path lengths
    all_paths = []
    roots = []
    tot_path_lengths = []

    if root_index is not None:
        soma_d = np.linalg.norm(mesh.vertices - mesh.vertices[root_index], axis=1)
        is_soma_pt = np.arange(len(mesh.vertices)) == root_index
    elif soma_pt is not None:
        soma_d = mesh.vertices - soma_pt.reshape(1, 3)
        soma_d = np.linalg.norm(soma_d, axis=1)
        is_soma_pt = soma_d < soma_thresh
    else:
        is_soma_pt = None
        soma_d = None
    # is_soma_pt = None
    # soma_d = None

    # loop over the components
    for k in range(n_components):
        if comp_counts[k] > cc_vertex_thresh:

            # find the root using a soma position if you have it
            # it will fall back to a heuristic if the soma
            # is too far away for this component
            root, root_ds, pred, valid = setup_root(
                mesh, is_soma_pt, soma_d, labels == k
            )
            # run teasar on this component
            teasar_output = mesh_teasar(
                mesh,
                root=root,
                root_ds=root_ds,
                root_pred=pred,
                valid=valid,
                invalidation_d=invalidation_d,
                return_map=return_map,
            )
            if return_map is False:
                paths, path_lengths = teasar_output
            else:
                paths, path_lengths, mesh_to_skeleton_map_single = teasar_output
                mesh_to_skeleton_map[
                    ~np.isnan(mesh_to_skeleton_map_single)
                ] = mesh_to_skeleton_map_single[~np.isnan(mesh_to_skeleton_map_single)]

            if len(path_lengths) > 0:
                # collect the results in lists
                tot_path_lengths.append(path_lengths)
                all_paths.append(paths)
                roots.append(root)

    if return_map:
        return all_paths, roots, tot_path_lengths, mesh_to_skeleton_map
    else:
        return all_paths, roots, tot_path_lengths


def setup_root(mesh, is_soma_pt=None, soma_d=None, is_valid=None):
    """ function to find the root index to use for this mesh """
    if is_valid is not None:
        valid = np.copy(is_valid)
    else:
        valid = np.ones(len(mesh.vertices), bool)
    assert len(valid) == mesh.vertices.shape[0]

    root = None
    # soma mode
    if is_soma_pt is not None:
        # pick the first soma as root
        assert len(soma_d) == mesh.vertices.shape[0]
        assert len(is_soma_pt) == mesh.vertices.shape[0]
        is_valid_root = is_soma_pt & valid
        valid_root_inds = np.where(is_valid_root)[0]
        if len(valid_root_inds) > 0:
            min_valid_root = np.nanargmin(soma_d[valid_root_inds])
            root = valid_root_inds[min_valid_root]
            root_ds, pred = sparse.csgraph.dijkstra(
                mesh.csgraph, directed=False, indices=root, return_predecessors=True
            )
        else:
            start_ind = np.where(valid)[0][0]
            root, target, pred, dm, root_ds = utils.find_far_points(
                mesh, start_ind=start_ind
            )
        valid[is_soma_pt] = False

    if root is None:
        # there is no soma close, so use far point heuristic
        start_ind = np.where(valid)[0][0]
        root, target, pred, dm, root_ds = utils.find_far_points(
            mesh, start_ind=start_ind
        )
    valid[root] = False
    assert np.all(~np.isinf(root_ds[valid]))
    return root, root_ds, pred, valid


def mesh_teasar(
    mesh,
    root=None,
    valid=None,
    root_ds=None,
    root_pred=None,
    soma_pt=None,
    soma_thresh=7500,
    invalidation_d=10000,
    return_timing=False,
    return_map=False,
    exclude_edges_sigma=None,
):
    """core skeletonization function used to skeletonize a single component of a mesh"""
    # if no root passed, then calculation one
    if root is None:
        root, root_ds, root_pred, valid = setup_root(
            mesh, soma_pt=soma_pt, soma_thresh=soma_thresh
        )
    # if root_ds have not be precalculated do so
    if root_ds is None:
        root_ds, root_pred = sparse.csgraph.dijkstra(
            mesh.csgraph, False, root, return_predecessors=True
        )
    # if certain vertices haven't been pre-invalidated start with just
    # the root vertex invalidated
    if valid is None:
        valid = np.ones(len(mesh.vertices), bool)
        valid[root] = False
    else:
        if len(valid) != len(mesh.vertices):
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

    with tqdm(total=total_to_visit, disable=True) as pbar:
        # keep looping till all vertices have been invalidated
        while np.sum(valid) > 0:
            k += 1
            t = time.time()
            # find the next target, farthest vertex from root
            # that has not been invalidated
            target = np.nanargmax(root_ds * valid)
            if np.isinf(root_ds[target]):
                raise Exception("target cannot be reached")
            time_arrays[0].append(time.time() - t)

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
            max_path_length = root_ds[target] - root_ds[max_branch]

            # calculate the shortest path to that vertex
            # from all other vertices
            # up till the distance to the root
            ds, pred_t = sparse.csgraph.dijkstra(
                mesh.csgraph,
                False,
                target,
                limit=max_path_length,
                return_predecessors=True,
            )

            # pick out the vertex that has already been visited
            # which has the shortest path to target
            min_node = np.argmin(ds[visited_nodes])
            # reindex to get its absolute index
            branch = visited_nodes[min_node]
            # this is in the index of the point on the skeleton
            # we want this branch to connect to
            time_arrays[1].append(time.time() - t)

            t = time.time()
            # get the path from the target to branch point
            path = utils.get_path(target, branch, pred_t)
            visited_nodes += path[0:-1]
            # record its length
            assert ~np.isinf(ds[branch])
            path_lengths.append(ds[branch])
            # record the path
            paths.append(path)
            time_arrays[2].append(time.time() - t)

            t = time.time()
            # get the distance to all points along the new path
            # within the invalidation distance
            dm, _, sources = sparse.csgraph.dijkstra(
                mesh.csgraph,
                False,
                path,
                limit=invalidation_d,
                min_only=True,
                return_predecessors=True,
            )
            time_arrays[3].append(time.time() - t)

            t = time.time()
            # all such non infinite distances are within the invalidation
            # zone and should be marked invalid
            nodes_to_update = ~np.isinf(dm)
            marked = np.sum(valid & ~np.isinf(dm))
            if return_map == True:
                new_sources_closer = (
                    dm[nodes_to_update] < mesh_to_skeleton_dist[nodes_to_update]
                )
                mesh_to_skeleton_map[nodes_to_update] = np.where(
                    new_sources_closer,
                    sources[nodes_to_update],
                    mesh_to_skeleton_map[nodes_to_update],
                )
                mesh_to_skeleton_dist[nodes_to_update] = np.where(
                    new_sources_closer,
                    dm[nodes_to_update],
                    mesh_to_skeleton_dist[nodes_to_update],
                )

            valid[~np.isinf(dm)] = False

            # print out how many vertices are still valid
            pbar.update(marked)
            time_arrays[4].append(time.time() - t)
    # record the total time
    dt = time.time() - start

    out_tuple = (paths, path_lengths)
    if return_map:
        out_tuple = out_tuple + (mesh_to_skeleton_map,)
    if return_timing:
        out_tuple = out_tuple + (time_arrays, dt)

    return out_tuple


def smooth_graph(values, edges, mask=None, neighborhood=2, iterations=100, r=0.1):
    """smooths a spatial graph via iterative local averaging
    calculates the average value of neighboring values
    and relaxes the values toward that average

    Parameters
    ----------
    values : numpy.array
        a NxK numpy array of values, for example xyz positions
    edges : numpy.array
        a Mx2 numpy array of indices into values that are edges
    mask : numpy.array
        NOT yet implemented
        optional N boolean vector of values to mask
        the vert locations.  the result will return a result at every vert
        but the values that are false in this mask will be ignored and not
        factored into the smoothing.
    neighborhood : int
        an integer of how far in the graph to relax over
        as being local to any vertex (default = 2)
    iterations : int
        number of relaxation iterations (default = 100)
    r : float
        relaxation factor at each iteration
        new_vertex = (1-r)*old_vertex*mask + (r+(1-r)*(1-mask))*(local_avg)
        default = .1

    Returns
    -------
    np.array
        new_verts, a Nx3 list of new smoothed vertex positions

    """
    N = len(values)
    E = len(edges)

    # setup a sparse matrix with the edges
    sm = sparse.csc_matrix((np.ones(E), (edges[:, 0], edges[:, 1])), shape=(N, N))

    # an identity matrix
    eye = sparse.csc_matrix(
        (np.ones(N, dtype=np.float32), (np.arange(0, N), np.arange(0, N))), shape=(N, N)
    )
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
    neighbors = 1 / neighbors
    C = C.multiply(neighbors)
    # convert back to csc
    C = C.tocsc()

    # multiply weights by relaxation term
    C *= r

    # construct relaxation matrix, adding on identity with complementary weight
    A = C + (1 - r) * eye

    # make a copy of original vertices to no destroy inpuyt
    new_values = np.copy(values)

    # iteratively relax the vertices
    for i in range(iterations):
        new_values = A * new_values
    return new_values


def soma_via_sphere(soma_pt, verts, edges, soma_d_thresh):
    """Get indices within soma_d_thresh of a soma_pt. Exclude vertices that left and come back."""
    closest_soma_ind = np.argmin(np.linalg.norm(verts - soma_pt, axis=1))
    close_inds = np.linalg.norm(verts - soma_pt, axis=1) < soma_d_thresh
    orig_graph = utils.create_csgraph(verts, edges, euclidean_weight=False)
    speye = sparse.diags(close_inds.astype(int))
    _, compids = sparse.csgraph.connected_components(orig_graph * speye)
    return np.flatnonzero(compids[closest_soma_ind] == compids), soma_d_thresh


def soma_via_branch_starts(
    sk,
    mesh,
    soma_pt,
    rs,
    search_radius=20000,
    fallback_radius=15000,
    cutoff_threshold=0.2,
    min_cutoff=0.1,
    dynamic_range=1,
    dynamic_threshold=False,
):
    """Runs down paths into the soma region and finds onset of each branch."""

    is_close = np.linalg.norm(sk.vertices - soma_pt, axis=1) < search_radius
    is_close_fallback = np.linalg.norm(sk.vertices - soma_pt, axis=1) < fallback_radius

    # Find segments that emerge from the soma region
    close_segs = []
    for seg in sk.segments:
        seg = seg[np.argsort(sk.distance_to_root[seg])]
        if is_close[seg[0]]:
            if np.all(is_close[sk.path_to_root(seg[0])]):
                close_segs.append(seg)
    close_seg_inds = np.concatenate(close_segs)
    close_inds = close_seg_inds[is_close[close_seg_inds]]

    # From those segments, find the tips that come out of the search_radius
    is_close_specific = np.full(sk.n_vertices, False)
    is_close_specific[close_inds] = True
    close_parent_edges = sk.edges[is_close_specific[sk.edges[:, 1]]]
    tip_inds = close_parent_edges[~is_close_specific[close_parent_edges[:, 0]], 0]

    rs = rs[close_inds]
    rs_long = np.full(sk.n_vertices, np.nan)
    rs_long[close_inds] = rs
    sk.reroot(close_inds[np.argmin(np.abs(rs - np.percentile(rs, 98)))])

    # Fit a logistic curve
    def log_func(x, h, k, xh, a):
        return h / (1 + np.exp(-k * (xh - x))) + a

    all_params = []
    soma_votes = []

    for tip_ind in tip_inds:
        ptr = sk.path_to_root(tip_ind)
        path_inds = ptr[1:]
        xdata = sk.distance_to_root[path_inds] / 1000
        ydata = rs_long[path_inds] / 1000
        good_rows = np.invert(np.logical_or(np.isnan(ydata), np.isinf(ydata)))
        ydata = ydata[good_rows]
        xdata = xdata[good_rows]
        ydata_filt = np.maximum.accumulate(signal.medfilt(ydata, 21))

        try:
            # sig = ydata_filt * log_func( (np.max(xdata)-xdata), 1, 2, 3, 1)
            sig = np.where(ydata_filt < 2, 2, ydata_filt) * log_func(
                (np.max(xdata) - xdata), 2, 1, 5, 1
            )
            params, _ = optimize.curve_fit(
                log_func,
                xdata,
                ydata_filt,
                sigma=sig,
                bounds=([0, 0.5, 0, 0], [np.inf, 5, np.inf, np.inf]),
                p0=(10, 1, 10, 1),
                method="trf",
            )
            all_params.append(params)
            if dynamic_threshold:
                cutoff_threshold_eff = np.min(
                    [
                        cutoff_threshold,
                        min_cutoff
                        + (cutoff_threshold - min_cutoff)
                        * np.max([0, (params[1] - 0.5) / dynamic_range]),
                    ]
                )
            else:
                cutoff_threshold_eff = cutoff_threshold

            def f(x):
                return log_func(x, *params) - (
                    params[3] + cutoff_threshold_eff * (params[0] - params[3])
                )

            opt_sol = optimize.root_scalar(f, bracket=[0, xdata.max()])
            if opt_sol.converged:
                root = opt_sol.root
                use_fallback = False
            else:
                use_fallback = True
        except:
            use_fallback = True
            all_params.append(None)

        if not use_fallback:
            base_path_ind = np.argmin(np.abs(xdata - root))
        else:
            d_to_soma_pt = np.linalg.norm(sk.vertices[path_inds] - soma_pt, axis=1)
            base_path_ind = np.argmin(np.abs(d_to_soma_pt - fallback_radius))

        soma_vote = np.full(sk.n_vertices, np.nan)
        soma_vote[path_inds[base_path_ind:]] = 1
        soma_vote[path_inds[:base_path_ind]] = 0
        soma_votes.append(soma_vote)

    # Any tips whose path to root is entirely in the close zone also vote as as 'somatic'
    for ep in sk.end_points[is_close[sk.end_points]]:
        ptr = sk.path_to_root(ep)
        if np.all(is_close[ptr]):
            soma_vote = np.full(sk.n_vertices, np.nan)
            soma_vote[ptr] = 1
            soma_vote[ptr[is_close_fallback[ptr]]] = np.inf
            soma_votes.append(soma_vote)

    # Get soma region
    soma_votes = np.vstack(soma_votes)
    num_votes = len(soma_votes) - np.sum(np.isnan(soma_votes), axis=0)
    num_yes = np.nansum(soma_votes, axis=0)

    with np.errstate(all="ignore"):
        is_soma = (num_yes / num_votes) > 0.5

    last_nonsoma = []
    for tip_ind in tip_inds:
        ptr = sk.path_to_root(tip_ind)
        last_nonsoma.append(ptr[np.flatnonzero(np.diff(is_soma[ptr]) == 1)[0]])
    last_nonsoma = np.unique(last_nonsoma)

    keep_binds = []
    for bind in last_nonsoma:
        bind_ptr = sk.path_to_root(bind)
        if np.any(np.isin(last_nonsoma, bind_ptr[1:])):
            keep_binds.append(False)
        else:
            keep_binds.append(True)
    keep_binds = np.array(keep_binds)

    g = sk.cut_graph(last_nonsoma[keep_binds])
    _, comps = sparse.csgraph.connected_components(g)
    root_comp = comps[sk.root]
    return np.flatnonzero(comps == root_comp), np.nanmedian(rs_long[comps == root_comp])


def collapse_soma_skeleton(
    soma_verts,
    soma_pt,
    verts,
    edges,
    mesh_to_skeleton_map=None,
    collapse_index=None,
    return_filter=False,
    return_soma_ind=False,
):
    """function to adjust skeleton result to move root to soma_pt

    Parameters
    ----------
    soma_pt : numpy.array
        a 3 long vector of xyz locations of the soma (None to just remove duplicate )
    verts : numpy.array
        a Nx3 array of xyz vertex locations
    edges : numpy.array
        a Kx2 array of edges of the skeleton
    soma_d_thresh : float
        distance from soma_pt to collapse skeleton nodes
    mesh_to_skeleton_map : np.array
        a M long array of how each mesh index maps to a skeleton vertex
        (default None).  The function will update this as it collapses vertices to root.
    soma_mesh_indices : np.array
         a K long array of indices in the mesh that should be considered soma
         Any  skeleton vertex on these vertices will all be collapsed to root.
    return_filter : bool
        whether to return a list of which skeleton vertices were used in the end
        for the reduced set of skeleton vertices
    only_soma_component : bool
        whether to collapse only the skeleton connected component which is closest to the soma_pt
        (default True)
    return_soma_ind : bool
        whether to return which skeleton index that is the soma_pt

    Returns
    -------
    np.array
        verts, Px3 array of xyz skeleton vertices
    np.array
        edges, Qx2 array of skeleton edges
    (np.array)
        new_mesh_to_skeleton_map, returned if mesh_to_skeleton_map and soma_pt passed
    (np.array)
        used_vertices, if return_filter this contains the indices into the passed verts which the return verts is using
    int
        an index into the returned verts that is the root of the skeleton node, only returned if return_soma_ind is True

    """
    if soma_verts is not None:
        soma_pt_m = soma_pt.reshape(1, 3)
        if collapse_index is None:
            new_verts = np.vstack((verts, soma_pt_m))
            soma_i = verts.shape[0]
        else:
            new_verts = verts
            soma_i = collapse_index
            soma_verts = soma_verts[soma_verts != soma_i]

        edges_m = edges.copy()
        edges_m[np.isin(edges, soma_verts)] = soma_i

        simple_verts, simple_edges = utils.remove_unused_verts(new_verts, edges_m)
        good_edges = ~(simple_edges[:, 0] == simple_edges[:, 1])

        if mesh_to_skeleton_map is not None:
            consolidate_dict = {v: soma_i for v in soma_verts}
            new_index_dict, _ = utils.remap_dict(len(new_verts), consolidate_dict)
            new_index_dict[-1] = -1
            mesh_to_skeleton_map[np.isnan(mesh_to_skeleton_map)] = -1
            new_mesh_to_skeleton_map = fastremap.remap(
                mesh_to_skeleton_map, new_index_dict
            )

        output = [simple_verts, simple_edges[good_edges]]
        if mesh_to_skeleton_map is not None:
            output.append(new_mesh_to_skeleton_map)
        if return_filter:
            used_vertices = np.unique(edges_m.ravel())
            if collapse_index is None:
                # Remove the largest value which is soma_i
                used_vertices = used_vertices[:-1]
            output.append(used_vertices)
        if return_soma_ind:
            output.append(new_index_dict[soma_i])
        return output

    else:
        simple_verts, simple_edges = utils.remove_unused_verts(verts, edges)
        return simple_verts, simple_edges


def resample(sk, spacing, kind="linear", tip_length_ratio=0.25):
    """Resample a skeleton's vertices

    Parameters
    ----------
    sk : Skeleton
        Input skeleton file with a skeleton
    spacing : numeric
        Desired spacing in nanometers
    kind : str, optional
        Type of interpolation to use when resampling. Options follow scipy.interpolate.interp1d. By default "linear"
    tip_length_ratio : float, optional
        The ratio of spacing to branch tip length that a branch tip must have in order to be included in the final skeleton
        for example: spacing is 10 and branch length is 8. do you want to include that final 8 length tip?
        then perhaps consider a tip_length_ratio of .75, by default 0.25

    Returns
    -------
    Skeleton
        New skeleton with resampled vertices.

    resample_map
        Array where the ith index corresponds to the ith vertex of the resampled skeleton and the value
        is the associated index in the original skeleton. To assign vertices, we assign a "domain" to each
        vertex in the original skeleton that is halfway between the vertex and its neighbors. Resampled
        vertices that fall within that domain (based on topology and distance-to-root) are then associated
        with the original vertex.
    """
    path_counter = 0
    branch_d = {}
    vert_list = []
    edge_list = []
    output_map_list = []

    for path in sk.cover_paths:
        new_verts, new_edges, output_map_path, branch_d = skeleton_utils.resample_path(
            path,
            sk,
            path_counter,
            spacing,
            kind,
            tip_length_ratio,
            branch_d,
        )
        vert_list.append(new_verts)
        edge_list.append(new_edges)
        output_map_list.append(output_map_path)
        path_counter += len(new_verts)

    new_verts = np.vstack(vert_list)
    new_edges = np.vstack(edge_list)
    resample_map = np.concatenate(output_map_list)

    return (
        Skeleton(
            new_verts,
            new_edges,
            root=branch_d[int(sk.root)],
            remove_zero_length_edges=False,
        ),
        resample_map,
    )