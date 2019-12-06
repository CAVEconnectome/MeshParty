import numpy as np
from scipy import sparse

def filter_close_to_line(mesh, line_end_pts, line_dist_th, axis=1, endcap_buffer=0, sphere_ends=False):
    '''
    Given a mesh and a line segment defined by two end points, make a filter
    leaving only those nodes within a certain distance of the line segment in
    a plane defined by a normal axis (e.g. the y axis defines distances in the
    xy plane)

    Parameters
    ----------
    mesh : meshparty.trimesh_io.Mesh
        Trimesh-like mesh with N vertices
    line_end_pts: numpy.array
        2x3 numpy array defining the two end points
    line_dist_th: numeric
        numeric, distance threshold
    axis: int
        integer 0-2. Defines which axis is normal to the plane in
        which distances is computed. optional, default 1 (y-axis).
    
    Returns
    -------
    numpy.array
        N-length boolean array

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
    """
    returns a boolean mask for vertices that are part of components in a size range

    Parameters
    ----------
    mesh : meshparty.trimesh_io.Mesh
        A Trimesh-like mesh with N vertices
    min_size : int
        the minimum number of vertices in compoment (default 0)
    max_size : int
        the maximum number of vertices in compoment (default infinity)

    Returns
    -------
    np.array
        N-length boolean array

    """
    cc, labels = sparse.csgraph.connected_components(mesh.csgraph, directed=False)
    uids, counts = np.unique(labels, return_counts=True)
    good_labels = uids[(counts>min_size)&(counts<=max_size)]
    return np.in1d(labels, good_labels)

def filter_largest_component(mesh):
    """ returns a boolean mask for vertices that are part of the largest component

    Parameters
    ----------
    mesh : meshparty.trimesh_io.Mesh
        A Trimesh-like mesh with N vertices

    Returns
    -------
    np.array
        N-length boolean array

    """
    cc, labels = sparse.csgraph.connected_components(mesh.csgraph)
    uids, counts = np.unique(labels, return_counts=True)
    max_label = np.argmax(counts)
    return labels==max_label


def filter_spatial_distance_from_points(mesh, pts, d_max):
    """
    returns a boolean mask for vertices near a set of points

    Parameters
    ----------
    mesh : meshparty.trimesh_io.Mesh
        A Trimesh-like mesh with N vertices
    pts : numpy.array
        a Kx3 set of points
    d_max : float
        the maximum distance to points to include (same units as mesh.vertices)

    Returns
    -------
    np.array
        N-length boolean array

    """
    if type(pts)==list:
        pts=np.array(pts)
    if len(pts.shape)==1:
        assert(len(pts)==3)
        ds = np.linalg.norm(mesh.vertices-pts[np.newaxis,:], axis=1)
        return ds<d_max
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

    Parameters
    ----------
    mesh : meshparty.trimesh_io.Mesh
        A Trimesh-like mesh with N vertices
    pts_foci: numpy.array
        2x3 array with the two foci in 3d space.
    d_pad: float
        Extra padding of the threhold distance beyond the distance between foci.
    indices : iterator
        Instead of pts_foci, one can specify a len(2) list of two indices into the mesh.vertices 
        default None. Will override pts_foci.
    power : int
        what power to use in Minkowski-like metrics for distance metric.

    Returns
    -------
    np.array
        N-length boolean array

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
    