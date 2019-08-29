import numpy as np
from scipy import sparse
import trimesh
from . import utils


#function to find the path along the mesh between a point on the mesh and the skeleton (closest skeleton point)
#if the point is not on the mesh, the point used is the mesh vertex that is closest to the point

def point_to_skel_meshpath(mesh,sk,pt):

    #mesh indices of skeleton vertices:
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
