import numpy as np
from scipy import sparse
import trimesh
from . import utils


def point_to_skel_meshpath(mesh, sk, pt, filterpts=None):
    '''

    Given a mesh, a skeleton and a point,  find the path along the mesh between a point 
    on the mesh and the skeleton (closest skeleton point). If the point is not on the mesh, 
    the point used is the mesh vertex that is closest to the point.

    Parameters
    ----------
    mesh: meshparty.trimesh_io.Mesh
          Trimesh-like mesh with N vertices
    sk: meshparty.trimesh_io.Mesh
          Skeleton whose vertices are a subset of the vertices of mesh
    pt : 1 x 3 numpy.array
          Array specifying a point location
    filterpts: Bool array 
          Filter that was used to generate "mesh", 
          If "mesh" is a filtered version of the mesh that was used to generate the skeleton "sk".
          If sk was generated from mesh as is, then use filter=None.

    Returns
    -------

    path: int array
          Indices of vertices on mesh which trace the path from the point (pt) to the skeleton (sk)

    '''

    if 'mesh_index' in sk.vertex_properties:
        mesh_index = sk.vertex_properties['mesh_index']
    else:
        mesh_index = sk.mesh_index

    if filterpts is None:
        sk_inds = [val for i, val in enumerate(mesh_index) if not val == -1]
    else:
        validinds = np.where(filterpts)[0]
        # intersection of validinds and sk.vertex_properties['mesh_index']
        localskeletoninds = list(
            set(mesh_index) & set(validinds))
        sk_inds = [i for i, val in enumerate(
            validinds) if val in localskeletoninds]

    if len(sk_inds) < 1:
        return None
    else:
        closest_point, distance, tid = trimesh.proximity.closest_point(
            mesh, [[pt[0], pt[1], pt[2]]])
        pointindex = mesh.faces[tid][0][0]

        dm, preds, sources = sparse.csgraph.dijkstra(
            mesh.csgraph, False, [pointindex],
            min_only=True, return_predecessors=True)
        min_node = np.argmin(dm[sk_inds])

        path = utils.get_path(pointindex, sk_inds[min_node], preds)
        return path
