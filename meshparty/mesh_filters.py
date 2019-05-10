import numpy as np
from scipy import sparse

def filter_close_to_line(mesh, line_end_pts, line_dist_th, axis=1, endcap_buffer=0, sphere_ends=False):
    '''
    Given a mesh and a line segment defined by two end points, make a filter
    leaving only those nodes within a certain distance of the line segment in
    a plane defined by a normal axis (e.g. the y axis defines distances in the
    xy plane)

    :param mesh: Trimesh-like mesh with N vertices
    :param line_end_pts: 2x3 numpy array defining the two end points
    :param line_dist_th: numeric, distance threshold
    :param axis: integer 0-2. Defines which axis is normal to the plane in
                 which distances is computed. optional, default 1 (y-axis).
    :returns:  N-length boolean array
    '''
    line_pt_ord = np.argsort(line_end_pts[:,axis])
    ds = _dist_from_line( mesh.vertices, line_end_pts, axis)

    below_top = mesh.vertices[:,axis] > line_end_pts[line_pt_ord[0], axis] - endcap_buffer
    above_bot = mesh.vertices[:,axis] < line_end_pts[line_pt_ord[1], axis] + endcap_buffer
    is_close = (ds < line_dist_th) & above_bot & below_top

    if sphere_ends is True:
        near_a = np.linalg.norm(mesh.vertices - line_end_pts[0], axis=1) < line_dist_th
        near_b = np.linalg.norm(mesh.vertices - line_end_pts[1], axis=1) < line_dist_th
        end_cap = near_a|near_b        
        is_close = is_close|end_cap
    return is_close


def _dist_from_line(pts, line_end_pts, axis):
    ps = (pts[:, axis] - line_end_pts[0, axis]) / (line_end_pts[1, axis] - line_end_pts[0, axis])
    line_pts = np.multiply(ps[:,np.newaxis], line_end_pts[1] - line_end_pts[0]) + line_end_pts[0]
    ds = np.linalg.norm(pts - line_pts, axis=1)
    return ds


def filter_components_by_size(mesh, min_size=0, max_size=np.inf):
    cc, labels = sparse.csgraph.connected_components(mesh.csgraph, directed=False)
    uids, counts = np.unique(labels, return_counts=True)
    good_labels = uids[(counts>min_size)&(counts<=max_size)]
    return np.in1d(labels, good_labels)

def filter_largest_component(mesh):
    cc, labels = sparse.csgraph.connected_components(mesh.csgraph)
    uids, counts = np.unique(labels, return_counts=True)
    max_label = np.argmax(counts)
    return labels==max_label

# def filter_large_component(mesh, size_thresh=1000):
#     '''
#     Returns a mesh filter without any connected components less than a size threshold

#     :param mesh: Trimesh-like mesh with N vertices
#     :param size_thresh: Integer, min size of a component to keep. Optional, default=1000.
#     :returns: N-length boolean array
#     '''
#     cc, labels = sparse.csgraph.connected_components(mesh.csgraph, directed=False)
#     uids, counts = np.unique(labels, return_counts=True)
#     good_labels = uids[counts>size_thresh]
#     return np.in1d(labels, good_labels)


# def filter_small_component(mesh, size_thresh=1000):
#     '''
#     Returns a mesh filter without any connected components less than a size threshold

#     :param mesh: Trimesh-like mesh with N vertices
#     :param size_thresh: Integer, min size of a component to keep. Optional, default=1000.
#     :returns: N-length boolean array
#     '''
#     cc, labels = sparse.csgraph.connected_components(mesh.csgraph, directed=False)
#     uids, counts = np.unique(labels, return_counts=True)
#     good_labels = uids[counts<size_thresh]
#     return np.in1d(labels, good_labels)


def filter_spatial_distance_from_points(mesh, pts, d_max):
    close_enough = np.full((len(mesh.vertices), len(pts)), False)
    for ii, pt in enumerate(pts):
        ds = np.linalg.norm(mesh.vertices-pt, axis=1)
        close_enough[:,ii] = ds<d_max
    return np.any(close_enough, axis=1)


def filter_two_point_distance(mesh, pts_foci, d_pad, indices=None, power=1):
    '''
    Returns a boolean array of mesh points such that the sum of the distance from a
    point to each of the two foci are less than a constant. The constant is set by
    the distance between the two foci plus a user-specified padding. Optionally, use
    other Minkowski-like metrics (i.e. x^n + y^n < d^n where x and y are the distances
    to the foci.)
    :param mesh: Trimesh-like mesh with N vertices
    :param pts_foci: 2x3 np array with the two foci in 3d space.
    :param d_pad: Extra padding of the threhold distance beyond the distance between foci.
    :returns: N-length boolean array
    '''
    if indices is None:
        _, minds_foci = mesh.kdtree.query(pts_foci)
    else:
        minds_foci = np.array(indices)

    if len(minds_foci) != 2:
        print('One or both mesh points were not found')
        return None
    
    d_foci_to_all = sparse.csgraph.dijkstra(mesh.csgraph,
                                            indices=minds_foci,
                                            unweighted=False,
                                            )
    dmax = d_foci_to_all[0, minds_foci[1]] + d_pad

    if np.isinf(dmax):
        print('Points are not in the same mesh component')
        return None
    
    if power != 1:
        is_in_ellipse = np.sum(np.power(d_foci_to_all, power), axis=0) < np.power(dmax,power)
    else:
        is_in_ellipse = np.sum(d_foci_to_all, axis=0) < dmax

    return is_in_ellipse
    