from typing import Optional, Union
import fastremap

import numpy as np
import pandas as pd
from scipy import sparse


def remap_vertices_and_edges(
    id_list: np.ndarray,
    edgelist: np.ndarray,
):
    """Remap unique ids in a list to 0-N-1 range and remap edges accordingly."""

    id_map = {int(lid): ii for ii, lid in enumerate(id_list)}
    edgelist_new = fastremap.remap(
        edgelist,
        id_map,
    )
    return id_map, edgelist_new


def process_spatial_columns(col_names="pt", suffix="position"):
    """
    Process spatial column names into a standard format.
    """
    if isinstance(col_names, str):
        col_names = [
            f"{col_names}_{suffix}_x",
            f"{col_names}_{suffix}_y",
            f"{col_names}_{suffix}_z",
        ]
    return col_names


def process_vertices(
    vertices: Union[np.ndarray, pd.DataFrame],
    spatial_columns: Optional[list] = None,
    labels: Optional[Union[dict, pd.DataFrame]] = None,
    vertex_index: Optional[Union[str, np.ndarray]] = None,
):
    "Process vertices and labels into a DataFrame and column labels."
    if isinstance(vertices, np.ndarray) or isinstance(vertices, list):
        spatial_columns = ["x", "y", "z"]
        vertices = pd.DataFrame(np.array(vertices), columns=spatial_columns)

    if spatial_columns is None:
        if vertices.shape[1] != 3:
            raise ValueError(
                '"Vertices must have 3 columns for x, y, z coordinates if no spatial_columns are provided.'
            )
        spatial_columns = vertices.columns
    else:
        implicit_label_columns = list(
            vertices.columns[~vertices.columns.isin(spatial_columns)]
        )

    if isinstance(labels, dict):
        labels = pd.DataFrame(labels, index=vertices.index)
        if labels.shape[0] != vertices.shape[0]:
            raise ValueError("Labels must have the same number of rows as vertices.")
    elif labels is None:
        labels = pd.DataFrame(index=vertices.index)

    label_columns = list(labels.columns) + implicit_label_columns

    vertices = vertices.merge(
        labels,
        left_index=True,
        right_index=True,
        how="left",
    )
    if vertex_index is not None:
        vertices = vertices.set_index(vertex_index)
    return vertices, spatial_columns, label_columns


def build_csgraph(vertices, edges, euclidean_weight=True, directed=False):
    """
    Builds a csr graph from vertices and edges, with optional control
    over weights as boolean or based on Euclidean distance.
    """
    edges = edges[edges[:, 0] != edges[:, 1]]
    if euclidean_weight:
        xs = vertices[edges[:, 0]]
        ys = vertices[edges[:, 1]]
        weights = np.linalg.norm(xs - ys, axis=1)
        use_dtype = np.float32
    else:
        weights = np.ones((len(edges),)).astype(np.int8)
        use_dtype = np.int8

    if directed:
        edges = edges.T
    else:
        edges = np.concatenate([edges.T, edges.T[[1, 0]]], axis=1)
        weights = np.concatenate([weights, weights]).astype(dtype=use_dtype)

    csgraph = sparse.csr_matrix(
        (weights, edges),
        shape=[
            len(vertices),
        ]
        * 2,
        dtype=use_dtype,
    )

    return csgraph


def connected_component_slice(G, ind=None, return_boolean=False):
    """
    Gets a numpy slice of the connected component corresponding to a
    given index. If no index is specified, the slice is of the largest
    connected component.
    """
    _, labels = sparse.csgraph.connected_components(G)
    if ind is None:
        label_vals, cnt = np.unique(labels, return_counts=True)
        ind = np.argmax(cnt)
        label = label_vals[ind]
    else:
        label = labels[ind]

    if return_boolean:
        return labels == label
    else:
        return np.flatnonzero(labels == label)


def find_far_points_graph(mesh_graph, start_ind=None, multicomponent=False):
    """
    Finds the maximally far point along a graph by bouncing from farthest point
    to farthest point.
    """
    d = 0
    dn = 1

    if start_ind is None:
        if multicomponent:
            a = connected_component_slice(mesh_graph)[0]
        else:
            a = 0
    else:
        a = start_ind
    b = 1

    k = 0
    pred = None
    ds = None
    while 1:
        k += 1
        dsn, predn = sparse.csgraph.dijkstra(
            mesh_graph, False, a, return_predecessors=True
        )
        if multicomponent:
            dsn[np.isinf(dsn)] = -1
        bn = np.argmax(dsn)
        dn = dsn[bn]
        if dn > d:
            b = a
            a = bn
            d = dn
            pred = predn
            ds = dsn
        else:
            break

    return b, a, pred, d, ds
