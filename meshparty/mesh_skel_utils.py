import numpy as np
from scipy import sparse
import trimesh
from . import utils


def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 


def point_to_skel_meshpath(mesh,sk,pt,filter=None):
    '''

    Given a mesh, a skeleton and a point,  find the path along the mesh between a point on the mesh and the skeleton (closest skeleton point). If the point is not on the mesh, the point used is the mesh vertex that is closest to the point
    :param mesh: Trimesh-like mesh with N vertices
    :param sk: a skeleton whose vertices are a subset of the vertices of mesh
    :param pt : a length 3 array specifying a point location
    :param filter: a boolean array which is the filter that was used to generate "mesh", 
     if "mesh" is a filtered version of the mesh that was used to generate the skeleton "sk".
     If sk was generated from mesh as is, then use filter=None
    '''


    if filter is None:
    	t = sk.vertex_properties['mesh_index']
    	sk_inds = [val for i,val in enumerate(t) if not val == -1 ]
    else:
	validinds = np.where(filtpts)[0]
        localskeletoninds = list(set(sk.vertex_properties['mesh_index']) & set(validinds)) #intersection of validinds and sk.vertex_properties['mesh_index']
	sk_inds = [i for i, val in enumerate(validinds) if val in localskeletoninds]


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
