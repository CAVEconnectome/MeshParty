import numpy as np
from scipy import sparse
import trimesh
from . import utils


def point_to_skel_meshpath(mesh,sk,pt):
    '''

    Given a mesh, a skeleton and a point,  find the path along the mesh between a point on the mesh and the skeleton (closest skeleton point). If the point is not on the mesh, the point used is the mesh vertex that is closest to the point
    :param mesh: Trimesh-like mesh with N vertices
    :param sk: a skeleton whose vertices are a subset of the vertices of mesh
    :param pt : a length 3 array specifying a point location
    
    '''

    t = sk.vertex_properties['mesh_index']
    sk_inds = [val for i,val in enumerate(t) if not val == -1 ]

    if len(sk_inds) < 1:
        return None
    else:
        closest_point, distance, tid = trimesh.proximity.closest_point(mesh,[[pt[0], pt[1], pt[2] ]]) 
        pointindex = mesh.faces[tid][0][0]

        dm, preds, sources = sparse.csgraph.dijkstra(
                        mesh.csgraph, False, [pointindex], 
                        min_only=True, return_predecessors=True)
        min_node = np.argmin(dm[sk_inds])

        path = utils.get_path(pointindex, sk_inds[min_node], preds)
        return path
