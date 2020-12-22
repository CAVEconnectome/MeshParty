import networkx as nx
from meshparty.meshwork import Meshwork
from meshparty.trimesh_io import Mesh
import pandas as pd
import numpy as np
from scipy import sparse


def _build_multicut_graph(nrn):
    G = nx.from_scipy_sparse_matrix(nrn.mesh.csgraph)

    G.add_node('source')
    G.add_node('target')

    source_edges = [('source', ii, np.inf)
                    for ii in nrn.anno['st_df'].df.query('type == "s"').mesh_index.values]
    target_edges = [('target', ii, np.inf)
                    for ii in nrn.anno['st_df'].df.query('type == "t"').mesh_index.values]

    G.add_weighted_edges_from(source_edges)
    G.add_weighted_edges_from(target_edges)

    return G


def _multicut_partitions(G, nrn):
    _, partition = nx.minimum_cut(G, 'source', 'target', capacity='weight')

    part0 = list(partition[0].difference({'source', 'target'}))
    part1 = list(partition[1].difference({'source', 'target'}))

    return nrn.MeshIndex(part0).to_mesh_mask_base, nrn.MeshIndex(part1).to_mesh_mask_base


def _build_nrn_with_st_annos(mesh, source_points, target_points):
    if isinstance(source_points, np.ndarray):
        source_points = source_points.tolist()
    if isinstance(target_points, np.ndarray):
        target_points = target_points.tolist()
    nrn = Meshwork(mesh, voxel_resolution=[1, 1, 1])
    source_df = pd.DataFrame(data={'pt_position': source_points})
    source_df['type'] = 's'
    target_df = pd.DataFrame(data={'pt_position': target_points})
    target_df['type'] = 't'
    st_df = source_df.append(target_df, ignore_index=True)
    st_df['pt_position'] = np.vstack(st_df['pt_position'].values).tolist()

    nrn.add_annotations('st_df', st_df, point_column='pt_position', anchored=True, overwrite=True)
    return nrn


def _build_local_mask(nrn, initial_window):
    ds = sparse.csgraph.dijkstra(
        nrn.mesh.csgraph, indices=nrn.anno['st_df'].mesh_index, limit=initial_window)
    d_sq = ds[:, nrn.anno['st_df'].mesh_index]
    if np.any(np.isinf(d_sq.ravel())):
        raise ValueError(
            "Initial window is too low (default: 10000) or points are in different components")

    # Centers mask on the point with the lowest mean distance to other points
    ctr_ind = np.argmin(np.mean(d_sq, axis=0))
    ctr_pt = nrn.anno['st_df'].mesh_index[ctr_ind]
    local_mask = np.invert(np.isinf(sparse.csgraph.dijkstra(
        nrn.mesh.csgraph, indices=ctr_pt, limit=np.max(d_sq[ctr_ind])+1)))
    return local_mask


def _faces_to_keep(p1mask, p2mask, nrn):
    p1mask_full = nrn.mesh.filter_unmasked_boolean(p1mask)
    p1faces = np.all(p1mask_full[nrn.mesh.faces], axis=1)

    p2mask_full = nrn.mesh.filter_unmasked_boolean(p2mask)
    p2faces = np.all(p2mask_full[nrn.mesh.faces], axis=1)

    neither_mask = np.invert(np.logical_or(p1mask_full, p2mask_full))
    nfaces = np.any(neither_mask[nrn.mesh.faces], axis=1)
    good_faces = np.logical_or(np.logical_or(p1faces, p2faces), nfaces)
    return good_faces


def _add_expected_edges(G, new_mesh, p1mask, p2mask, local_network_mask, test_split=True):
    "Adds edges that were not included in the faces graph"
    G.remove_node('source')
    G.remove_node('target')

    new_mesh_filt = new_mesh.apply_mask(local_network_mask)
    p1s = new_mesh_filt.filter_unmasked_boolean(p1mask)
    p2s = new_mesh_filt.filter_unmasked_boolean(p2mask)

    # Make matrix without cross-partition edges
    Gorig = nx.to_scipy_sparse_matrix(G)
    ii, jj, dd = sparse.find(Gorig)
    keep11 = p1s[ii] & p1s[jj]
    keep22 = p2s[ii] & p2s[jj]
    keep_all = keep11 | keep22

    GsplitB = sparse.csr_matrix((dd[keep_all], (ii[keep_all], jj[keep_all]))).toarray() > 0

    Gnew = new_mesh_filt.csgraph.toarray()
    GnewB = Gnew > 0

    # Places where edge in expected Gmat but not in new mesh
    link_edges_to_add_rough = np.vstack(np.where(np.logical_and(GsplitB == True, GnewB == False))).T
    if len(link_edges_to_add_rough) > 0:
        link_edges_to_add = np.unique(
            [tuple(x) for x in np.sort(link_edges_to_add_rough, axis=1)], axis=0)

        link_edges_unmasked = new_mesh_filt.map_indices_to_unmasked(link_edges_to_add)
        new_mesh.link_edges = np.vstack(
            (new_mesh.link_edges, new_mesh.filter_unmasked_indices(link_edges_unmasked)))

    if test_split:
        if len(link_edges_to_add_rough) > 0:
            new_mesh_filt.link_edges = np.vstack((new_mesh_filt.link_edges, link_edges_to_add))

        ncomp = sparse.csgraph.connected_components(new_mesh_filt.csgraph)[0]
        if ncomp > 2:
            print('Warning: more than 2 local components after split')
        if ncomp == 1:
            print('Warning: Only 1 local component after split')

    return new_mesh


def mesh_multicut(mesh, source_points, target_points, initial_window=10000, return_masks=False):
    """Use multi-point source/target split to cut a minimal set of faces from a mesh.
    Warns if the split produces more than 2 graph components in a local cutout, although
    the end result may still be suitable.

    Parameters
    ----------
    mesh : Mesh object
        Mesh to split
    source_points : np.array
        Nx3 numpy array of Euclidean points on one side of the desired partition
    target_points : np.array
        Mx3 numpy array of Euclidean points on the other side of the desired partition
    initial_window : int or float, optional
        Search window for a point-to-point distance matrix, needs to be larger than the max distance between source and/or target points along the mesh, by default 10000
    return_masks : bool, optional
        If True, returns vertex masks that denote each partition. Default is False

    Returns
    -------
    split_mesh : Mesh object
        Mesh with the same vertices as the original, but faces split per the multicut.
    partition_mask_source : np.array
        Boolean mask with True for mesh nodes in source partition. Returned if return_masks
        is True.
    partition_mask_target : np.array
        Boolean mask with True for mesh nodes in source partition. Returned if return_masks
        is True.
    """
    nrn = _build_nrn_with_st_annos(mesh, source_points, target_points)

    local_network_mask = _build_local_mask(nrn, initial_window)
    curr_mask = nrn.mesh_mask

    nrn.apply_mask(local_network_mask)

    G = _build_multicut_graph(nrn)
    p1mask, p2mask = _multicut_partitions(G, nrn)

    nrn.reset_mask()
    nrn.apply_mask(curr_mask)

    keep_faces = _faces_to_keep(p1mask, p2mask, nrn)

    new_mesh = Mesh(vertices=nrn.mesh.vertices,
                    faces=nrn.mesh.faces[keep_faces], node_mask=nrn.mesh_mask, link_edges=nrn.mesh.link_edges)

    new_mesh = _add_expected_edges(G, new_mesh, p1mask, p2mask, local_network_mask)

    if return_masks:
        return new_mesh, p1mask, p2mask
    else:
        return new_mesh
