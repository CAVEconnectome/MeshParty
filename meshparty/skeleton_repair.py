import numpy as np
from meshparty.skeleton import Skeleton
from meshparty.skeletonize import smooth_graph
from meshparty import utils


def remove_duplicates(skel):
    """
    Remove duplicate vertices from a skeleton

    Parameters
    ----------
    skel: meshparty.skeleton.Skeleton
        The skeleton to process

    Returns
    -------
    deduped: meshparty.skeleton.Skeleton
        A copy of the original skeleton with duplicate vertices removed
    vertexinds: np.array
        The indices that survived de-duplication in the same order as
        the returned skeleton above
    """
    weights = utils.edge_weights(skel.vertices, skel.edges,
                                 euclidean_weight=True)

    # merging srcids -> dstids
    srcids = np.max(skel.edges[weights == 0], axis=1)
    dstids = np.min(skel.edges[weights == 0], axis=1)

    # masking vertex-based attributes
    vertexmask = np.ones((len(skel.vertices),), dtype=np.uint8)
    vertexmask[srcids] = 0
    vertexinds = np.flatnonzero(vertexmask)

    newverts = skel.vertices[vertexinds]
    newvertprops = {k: type(v)(np.array(v)[vertexinds])
                    for (k, v) in skel.vertex_properties.items()}

    # remapping vertex ids across other attributes
    remapping = np.zeros((len(skel.vertices),), dtype=np.uint32)
    remapping[vertexinds] = np.arange(len(newverts))
    remapping[srcids] = remapping[dstids]

    newedges = remapping[skel.edges[weights != 0]]
    newm2s = remapping[skel.mesh_to_skel_map]
    newroot = remapping[skel.root]

    newskel = Skeleton(vertices=newverts, edges=newedges,
                       mesh_to_skel_map=newm2s,
                       vertex_properties=newvertprops,
                       root=newroot)

    return newskel, vertexinds


def smooth_skeleton(skel, mask=None, neighborhood=2, iterations=100, r=0.1,
                    fix_root=True, fix_branchpoints=True, fix_endpoints=True):
    """
    Smoothing skeleton vertices. Adds a few extra controls on top of
    skeletonize.smooth_graph.

    Parameters
    ----------
    skel: meshparty.skeleton.Skeleton
        The skeleton to process
    mask: np.array
        A binary mask of vertices to smooth. Vertex indices that map
        to False will be fixed throughout smoothing on top of the other
        arguments below
    neighborhood: int
        The size of the neighborhood to use for smoothing. All nodes
        (except self) within this number of edges will be averaged
        to make a local average target for smoothing at each iteration.
        default = 2
    iterations: int
        The number of smoothing iterations to perform
        default = 100
    r : float
        Relaxation factor at each iteration
        v_{t+1} = r*(local_avg) + (1-r)*v_{t}  [mask == True]
        v_{t+1} = v_{t}                        [mask == False]
        default = 0.1

    Returns
    -------
    smoothed: meshparty.skeleton.Skeleton
        A copy of the original skeleton with smoothed vertices
    """
    mask = np.ones((len(skel.vertices),),
                   dtype=np.bool) if mask is None else mask

    if fix_root:
        mask[skel.root] = False

    if fix_branchpoints:
        mask[skel.branch_points] = False

    if fix_endpoints:
        mask[skel.end_points] = False

    newverts = smooth_graph(
                   skel.vertices, skel.edges, mask=mask,
                   neighborhood=neighborhood, iterations=iterations, r=r)

    return Skeleton(vertices=newverts, edges=skel.edges,
                    mesh_to_skel_map=skel.mesh_to_skel_map,
                    vertex_properties=skel.vertex_properties,
                    root=skel.root)
