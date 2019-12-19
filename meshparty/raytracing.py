from scipy import sparse, spatial
import numpy as np
import trimesh.ray
from trimesh.ray import ray_pyembree
import logging
import multiwrapper.multiprocessing_utils as mu

def vogel_disk_sampler(num_points, radius=1):
    "Gives points that sample a unit disk as best as possible"
    golden_angle = np.pi * (3 - np.sqrt(5))
    thetas = golden_angle * np.arange(num_points)
    radii = radius * np.sqrt(np.arange(num_points)/num_points)

    xs = radii * np.cos(thetas)
    ys = radii * np.sin(thetas)
    
    return xs, ys

def unit_vector_sampler(num_points, widest_angle=np.pi/3):
    if np.abs(widest_angle) > np.pi:
        print('')
        return None
    xs, ys = vogel_disk_sampler(num_points, radius=1)
    zs = 1/np.tan(widest_angle) * np.ones(num_points)
    Vs = np.vstack((xs, ys, zs)).T
    return Vs / np.linalg.norm(Vs, axis=1)[:, np.newaxis]

def Rx( ang ):
    return np.array([[1, 0, 0],
                     [0, np.cos(ang), -np.sin(ang)],
                     [0, np.sin(ang), np.cos(ang)]])

def Ry( ang ):
    return np.array([[np.cos(ang), 0, np.sin(ang)],
                     [0,           1,       0    ],
                     [-np.sin(ang),0, np.cos(ang)]])

def Rz( ang ):
    return np.array([[np.cos(ang), -np.sin(ang), 0],
                     [np.sin(ang), np.cos(ang), 0],
                     [0,  0, 1]])

def _rotated_cone(data):
    phi, theta, vs_raw = data
    Rtrans = np.dot(Rz(phi), Ry(theta))
    return np.dot(Rtrans, vs_raw.T).T
    
def oriented_vector_cones(center_vectors, num_points, widest_angle=np.pi/3, normalize=False):
    'Produces cones oriented along a list of center directions' 
    if normalize:
        cv_norm = center_vectors / np.linalg.norm(center_vectors, axis=1)[:, np.newaxis]
    else:
        cv_norm = center_vectors
        
    thetas = np.arccos(cv_norm[:, 2])
    phis = np.arctan2(cv_norm[:, 1], cv_norm[:, 0])
    
    vs_raw = unit_vector_sampler(num_points, widest_angle=widest_angle)
    
    data = []
    for phi, theta in zip(phis, thetas):
        data.append((phi, theta, vs_raw))
    vector_cones = mu.multiprocess_func(_rotated_cone, data)
    return vector_cones

def _angle_weighted_distance(ds, angles, weights):
    if len(ds)==0 or np.nansum(weights)==0:
        return np.nan
    
    med_angle = np.median(angles)
    std_angle = np.std(angles)
    min_angle = med_angle-std_angle
    max_angle = max(med_angle+std_angle, np.pi / 2)
    good_rows = np.logical_and(angles>=min_angle, angles<=max_angle)
    
    nanaverage = np.nansum(ds[good_rows] * weights[good_rows]) / np.nansum(weights[good_rows])
    return nanaverage


def _multi_angle_weighted_distance(data):
    ds, angles, weights = data
    return _angle_weighted_distance(ds, angles, weights)

def all_angle_weighted_distances(ds, angles, weights, rep_inds, inds):
    "Averages distance measures based on the inverse hit angle, removing outliers"
    data = []
    real_inds, slice_bnds = np.unique(rep_inds, return_index=True)
    
    ind_map = []
    for ii in range(len(real_inds)-1):
        row = slice(slice_bnds[ii], slice_bnds[ii+1])
        data.append((ds[row], angles[row], weights[row]))
        ind_map.append(real_inds[ii])
    row = slice(slice_bnds[-1], len(ds))
    data.append( (ds[row], angles[row], weights[row]) )
    ind_map.append(real_inds[-1])
        
    rs = mu.multiprocess_func(_multi_angle_weighted_distance, data)
    
    rs_out = np.nan * np.zeros(len(inds))
    rs_out[np.array(ind_map)] = rs
    
    return rs_out

def _compute_ray_vectors(mesh, mesh_inds, num_points, cone_angle):
    return np.vstack( oriented_vector_cones(-mesh.vertex_normals[mesh_inds], num_points, cone_angle) )

def sdf_distance(vertex_inds, mesh, num_points=30, cone_angle=np.pi/3):
    """Compute SDF (a distance approximation) using a cone of rays weighted by acuteness of the hit.
    
    Parameters
    ----------
    vertex_inds : list-like
        Mesh indices of vertices for which to compute SDF
    mesh : trimesh-like Mesh
        Mesh object for the SDF computation
    num_points : int, optional
        Number of rays to cast per point, by default 30
    cone_angle : float, optional
        Half-width of the angle subtended by rays, by default np.pi/3
    
    Returns
    -------
    numpy.array
        N-length array with the SDF for the given vertex indices.
    """
    start = (mesh.vertices-mesh.vertex_normals)[vertex_inds,:]
    rep_inds = np.concatenate([ii*np.ones(num_points, dtype=int) for ii in range(start.shape[0])])
    starts = start[rep_inds]
    
    vs = _compute_ray_vectors(mesh, vertex_inds, num_points, cone_angle)
    
    ray_inter = ray_pyembree.RayMeshIntersector(mesh)
    rtrace = ray_inter.intersects_location(starts, vs, multiple_hits=False)
    
    hit_rows = rtrace[1]
    ds = np.linalg.norm(rtrace[0] - starts[hit_rows], axis=1)
    angles = np.arccos(np.sum(mesh.face_normals[rtrace[2]] * vs[hit_rows], axis=1))
    weights = 1/angles
    good_rows = np.isfinite(weights)
    
    rs = all_angle_weighted_distances(ds[good_rows], angles[good_rows], weights[good_rows], rep_inds[hit_rows[good_rows]], vertex_inds)
    return rs

def ray_trace_distance(vertex_inds, mesh, max_iter=10, rand_jitter=0.001, verbose=False, ray_inter=None):
    '''
    Compute distance to opposite side of the mesh for specified vertex indices on the mesh using a single ray.

    Parameters
    ----------
    vertex_inds : np.array
        a K long set of indices into the mesh.vertices that you want to perform ray tracing on
    mesh : :obj:`meshparty.trimesh_io.Mesh`
        mesh to perform ray tracing on
    max_iter : int
        maximum retries to attempt in order to get a proper sdf measure (default 10)
    rand_jitter : float
        the amplitude of gaussian jitter on the vertex normal to add on each iteration (default .001)
    verbose : bool
        whether to print debug statements (default False)
    ray_inter: ray_pyembree.RayMeshIntersector
        a ray intercept object pre-initialized with a mesh, in case y ou are doing this many times
        and want to avoid paying initialization costs. (default None) will initialize it for you

    Returns
    -------
    np.array
        rs, a K long array of sdf values. rays with no result after max_iters will contain zeros.

    '''
    if not trimesh.ray.has_embree:
        logging.warning("calculating rays without pyembree, conda install pyembree for large speedup")

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
