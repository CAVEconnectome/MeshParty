# Additional properties for layers with edges
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Tuple, Union, Self

from scipy import sparse
from . import utils

import copy
import fastremap
import numpy as np
import pandas as pd
from .sync_classes import (
    MorphSync,
    PointSyncWork,
    GraphSyncWork,
    SkeletonSyncWork,
    Facet,
    Link,
)
from . import graph_functions as gf

SKEL_LAYER_NAME = "skeleton"
GRAPH_LAYER_NAME = "graph"
MESH_LAYER_NAME = "mesh"


class EdgeMixin(ABC):
    _csgraph = None
    _csgraph_binary = None

    @property
    def edges(self) -> np.ndarray:
        return self.layer.edges

    @property
    def edge_df(self) -> pd.DataFrame:
        return self.layer.edges_df

    @property
    def edges_positional(self) -> np.ndarray:
        return self.layer.edges_positional

    def _map_edges_to_index(self, edges, vertex_indices):
        index_map = {ii: v for ii, v in enumerate(vertex_indices)}
        return fastremap.remap(edges, index_map)

    @property
    def csgraph(self):
        if self._csgraph is None:
            self._csgraph = utils.build_csgraph(
                self.vertices,
                self.edges_positional,
                euclidean_weight=True,
                directed=True,
            )
        return self._csgraph

    @property
    def csgraph_binary(self):
        if self._csgraph_binary is None:
            self._csgraph_binary = utils.build_csgraph(
                self.vertices,
                self.edges_positional,
                euclidean_weight=False,
                directed=True,
            )
        return self._csgraph_binary

    @property
    def csgraph_undirected(self):
        return self.csgraph + self.csgraph.T

    @property
    def csgraph_binary_undirected(self):
        return self.csgraph_binary + self.csgraph_binary.T

    def _reset_derived_properties(self):
        self._csgraph = None
        self._csgraph_binary = None

    def distance_between(
        self,
        sources: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
        positional=False,
        limit: Optional[float] = None,
    ) -> np.ndarray:
        """
        Get the distance between two sets of vertices in the skeleton.

        Parameters
        ----------
        sources : Optional[np.ndarray]
            The source vertices. If None, all vertices are used.
        targets : Optional[np.ndarray]
            The target vertices. If None, all vertices are used.
        positional: bool
            Whether the input vertices are positional (i.e., masks or indices).
            Must be the same for sources and targets.
        limit: Optional[float]
            The maximum distance to consider in the graph distance lookup. If None, no limit is applied.
            Distances above this will be set to infinity.

        Returns
        -------
        np.ndarray
            The distance between each source and target vertex, of dimensions len(sources) x len(targets).
        """
        # Sources must be positional for the dijkstra
        sources, positional_sources = self._vertices_to_positional(sources, positional)
        targets, positional_targets = self._vertices_to_positional(targets, positional)
        if positional_sources != positional_targets:
            raise ValueError(
                "sources and targets must both be positional or both be indices. Masks are implicitly positional."
            )
        if limit is None:
            limit = np.inf
        return gf.source_target_distances(
            sources=sources,
            targets=targets,
            csgraph=self.csgraph_undirected,
            limit=limit,
        )

    def path_between(
        self,
        source: int,
        target: int,
        positional=False,
        as_vertices=False,
    ) -> np.ndarray:
        """
        Get the shortest path between two vertices in the skeleton.

        Parameters
        ----------
        source : int
            The source vertex.
        target : int
            The target vertex.
        positional: bool
            Whether the input vertices are positional (i.e., masks or indices).
            Must be the same for sources and targets.
        as_vertices: bool
            Whether to return the path as vertex IDs or 3d positions.

        Returns
        -------
        np.ndarray
            The shortest path between each source and target vertex, indices if positional is False, or nx3 array if `as_vertices` is True.
        """
        # Sources must be positional for the dijkstra
        st, positional_sources = self._vertices_to_positional(
            [source, target], positional
        )
        source = st[0]
        target = st[1]
        path_positional = gf.shortest_path(
            source=source,
            target=target,
            csgraph=self.csgraph_binary_undirected,
        )
        if positional and not as_vertices:
            return self.vertex_index[path_positional]
        else:
            if as_vertices:
                return self.vertices[path_positional]
            else:
                return path_positional


# General properties for layers with points
class PointMixin(ABC):
    def _setup_properties(
        self,
        name: str,
        morphsync: Optional[PointSyncWork] = None,
        vertices: Union[np.ndarray, pd.DataFrame] = None,
        spatial_columns: Optional[list] = None,
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        vertex_index: Optional[Union[str, np.ndarray]] = None,
    ):
        self._name = name
        if morphsync is None:
            self._morphsync = MorphSync()
        else:
            self._morphsync = morphsync
        vertices, spatial_columns, label_columns = utils.process_vertices(
            vertices=vertices,
            spatial_columns=spatial_columns,
            labels=labels,
            vertex_index=vertex_index,
        )
        self._spatial_columns = spatial_columns
        self._label_columns = label_columns
        return vertices, spatial_columns, label_columns

    def _setup_linkage(
        self,
        linkage: Optional[Link] = None,
    ):
        if linkage is not None:
            if linkage.source is None:
                linkage.source = self.layer_name
            elif linkage.target is None:
                linkage.target = self.layer_name
            if isinstance(linkage.mapping, str):
                linkage.mapping = (
                    self._morphsync._layers[linkage.source]
                    .nodes[linkage.mapping]
                    .values
                )
            self._process_linkage(linkage)

    def _vertices_to_positional(
        self,
        vertices: Optional[np.ndarray],
        positional: bool,
    ) -> Tuple[np.ndarray, bool]:
        """Map vertex index to positional indices whether inputs are positional, masks, indices."""
        if vertices is None:
            vertices = np.arange(self.n_vertices)
            positional = True
        else:
            vertices = np.array(vertices)
            if np.issubdtype(vertices.dtype, np.bool_):
                if len(vertices) != self.n_vertices:
                    raise ValueError(
                        "If vertices is a boolean array, it must have the same length as the number of vertices."
                    )
                vertices = np.flatnonzero(vertices)
                positional = True
            if not positional:
                vertices = fastremap.remap(vertices, self.vertex_index_map)
            vertices = np.array(vertices)
        return vertices, positional

    @property
    def name(self) -> str:
        return self._name

    @property
    def layer(self) -> PointSyncWork:
        return self._get_layer(self.layer_name)

    def _get_layer(self, layer_name: str) -> Facet:  # type: ignore
        return self._morphsync._layers[layer_name]

    @property
    def vertices(self) -> np.ndarray:
        return self.layer.vertices

    @property
    def vertex_df(self) -> pd.DataFrame:
        return self.layer.vertices_df

    @property
    def vertex_index(self) -> pd.Index:
        return np.array(self.layer.vertices_index)

    @property
    def vertex_index_map(self) -> dict:
        map = {int(v): ii for ii, v in enumerate(self.vertex_index)}
        map[-1] = -1
        return map

    @property
    def nodes(self) -> pd.DataFrame:
        return self.layer.nodes

    @property
    def spatial_columns(self) -> list:
        return self._spatial_columns

    @property
    def label_names(self) -> list:
        return self._label_columns

    @property
    def labels(self) -> pd.DataFrame:
        return self.nodes[self.label_names]

    @property
    def n_vertices(self) -> int:
        return self.layer.n_vertices

    def get_label(self, key) -> np.ndarray:
        return self.labels[key].values

    def add_label(
        self,
        label: Union[list, np.ndarray, dict, pd.DataFrame],
        name: Optional[str] = None,
    ):
        if isinstance(label, list) or isinstance(label, np.ndarray):
            label = pd.DataFrame(label, index=self.vertex_index, columns=[name])
        elif isinstance(label, dict):
            label = pd.DataFrame(label, index=self.vertex_index)
        elif isinstance(label, pd.DataFrame):
            label = label.loc[self.vertex_index]
        else:
            raise ValueError("Label must be a list, np.ndarray, dict or pd.DataFrame.")

        if label.shape[0] != self.n_vertices:
            raise ValueError("Label must have the same number of rows as vertices.")
        if np.any(label.columns.isin(self.nodes.columns)):
            raise ValueError('"Label name already exists in the nodes DataFrame.")')

        self._morphsync._layers[self.layer_name].nodes = self.nodes.merge(
            label,
            left_index=True,
            right_index=True,
            how="left",
            validate="1:1",
        )
        self._label_columns += list(label.columns)

    def map_index_to_layer(
        self, layer: str, source_index=None, positional: bool = False
    ) -> Optional[int]:
        """Map each vertex index from the current layer to the specified layer.

        Parameters
        ----------
        layer : str
            The target layer to map the index to.
        source_index : Optional[np.ndarray]
            The source index to map from. If None, all vertices are used. Can also be a boolean array.
        positional : bool
            Whether to treat source_index and mapped index as positional (i_th element of the array) or as a dataframe index.

        Returns
        -------
        Optional[int]
            The mapped index in the target layer, or None if not found.
            If `positional` is True, the mapping is based on the position of the vertices not the dataframe index.
        """

        if source_index is None:
            source_index = self.vertex_index
        source_index = np.array(source_index)
        if positional or np.issubdtype(source_index.dtype, np.bool):
            source_index = self.vertex_index[source_index]
        if layer in self._morphsync._layers:
            mapping = self._morphsync.get_mapping(
                source=self.layer_name, target=layer, source_index=source_index
            )
            if positional:
                return fastremap.remap(
                    mapping,
                    {
                        int(k): ii
                        for ii, k in enumerate(
                            np.array(
                                self._morphsync._layers[layer].vertices_index.values
                            )
                        )
                    },
                )
            else:
                return mapping
        else:
            raise ValueError(f"Layer '{layer}' does not exist.")

    def map_mask_to_layer(
        self,
        layer: str,
        source_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Map a boolean mask on the current layer to the vertices of the specified layer.

        Parameters
        ----------
        layer: str
            The target layer to map the mask to.
        source_mask: np.ndarray
            The boolean mask to map from.

        Returns
        -------
        np.ndarray
            The mapped boolean mask for the target layer.
        """
        source_indices = self.vertex_index[np.array(source_mask)]
        if layer in self._morphsync._layers:
            mapping = self._morphsync.get_mapping(
                source=self.layer_name, target=layer, source_index=source_indices
            )
            target_layer = self._morphsync._layers[layer]
            return target_layer.vertices_index.isin(mapping)
        else:
            raise ValueError(f"Layer '{layer}' does not exist.")

    def _mask_morphsync(
        self,
        mask: Optional[np.ndarray] = None,
    ):
        if mask is not None:
            mask = np.array(mask)
            if len(mask) == self.n_vertices and np.issubdtype(mask.dtype, np.bool_):
                mask = mask.astype(bool)
            else:
                mask = self.vertex_index.isin(mask)
        else:
            mask = self.vertex_index

        return self._morphsync.apply_mask(
            layer_name=self.layer_name,
            mask=mask,
        )

    def _process_linkage(
        self,
        full_link: Link,
    ):
        source_layer = self._get_layer(full_link.source)
        target_layer = self._get_layer(full_link.target)

        if len(full_link.mapping) == source_layer.n_vertices:
            self._morphsync.add_link(
                source=full_link.source,
                target=full_link.target,
                mapping=full_link.mapping_to_index(target_layer.nodes),
            )
        else:
            raise ValueError("Mapping must have the same number of rows as vertices.")

    @abstractmethod
    def apply_mask(
        self,
        mask: Optional[np.ndarray] = None,
        new_morphsync: Optional[MorphSync] = None,
    ):
        pass

    def _register_meshworksync(self, mws):
        self._mws = mws


class GraphSync(PointMixin, EdgeMixin):
    layer_name = GRAPH_LAYER_NAME

    def __init__(
        self,
        name: str,
        vertices: Union[np.ndarray, pd.DataFrame],
        edges: Union[np.ndarray, pd.DataFrame],
        spatial_columns: Optional[list] = None,
        *,
        vertex_index: Optional[Union[str, np.ndarray]] = None,
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        morphsync: MorphSync = None,
        linkage: Optional[Link] = None,
    ):
        vertices, spatial_columns, labels = self._setup_properties(
            name=name,
            morphsync=morphsync,
            vertices=vertices,
            spatial_columns=spatial_columns,
            labels=labels,
            vertex_index=vertex_index,
        )
        if vertex_index:
            edges = self._map_edges_to_index(edges, vertices.index)
        self._morphsync.add_graph(
            graph=(vertices, edges),
            name=self.layer_name,
            spatial_columns=spatial_columns,
        )
        self._setup_linkage(linkage)
        self._mws = None

    def apply_mask(
        self,
        mask: Optional[np.ndarray] = None,
        new_morphsync: Optional[MorphSync] = None,
    ):
        if new_morphsync is None:
            new_morphsync = self._mask_morphsync(mask=mask)
        return self.__class__(
            name=self.name,
            vertices=new_morphsync.layers.loc[self.layer_name].layer.vertices_df,
            edges=new_morphsync.layers.loc[self.layer_name].layer.edges_df,
            spatial_columns=self.spatial_columns,
            labels=new_morphsync.layers.loc[self.layer_name].layer.nodes,
            morphsync=new_morphsync,
        )

    def __repr__(self) -> str:
        return f"GraphSync(name={self.name}, vertices={self.vertices.shape[0]}, edges={self.edges.shape[0]})"


class SkeletonSync(GraphSync):
    layer_name = SKEL_LAYER_NAME

    def __init__(
        self,
        name: str,
        vertices: Union[np.ndarray, pd.DataFrame],
        edges: Union[np.ndarray, pd.DataFrame],
        spatial_columns: Optional[list] = None,
        root: Optional[int] = None,
        *,
        vertex_index: Optional[Union[str, np.ndarray]] = None,
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        morphsync: MorphSync = None,
        linkage: Optional[dict] = None,
        inherited_properties: Optional[dict] = None,
    ):
        vertices, spatial_columns, labels = self._setup_properties(
            name=name,
            morphsync=morphsync,
            vertices=vertices,
            spatial_columns=spatial_columns,
            labels=labels,
            vertex_index=vertex_index,
        )
        if vertex_index:
            edges = self._map_edges_to_index(edges, vertices.index)
        self._morphsync.add_graph(
            graph=(vertices, edges),
            name=self.layer_name,
            spatial_columns=spatial_columns,
        )
        self._setup_linkage(linkage)

        self._root = self._infer_root(root)
        self._dag_cache = gf.DAGCache(
            root=self._root
        )  # Cache of properties associated with rooted skeletons
        self._dag_cache.parent_node_array = self._apply_root_to_edges(root)

        if inherited_properties is None:
            self._set_base_properties(
                base_properties={
                    "base_root": self.root,
                    "base_vertex_index": self.vertex_index,
                    "base_parent_array": self.parent_node_array,
                    "base_csgraph": self.csgraph,
                    "base_csgraph_binary": self.csgraph_binary,
                }
            )
        else:
            self._set_base_properties(base_properties=inherited_properties)
        self._mws = None

    @property
    def root(self) -> int:
        return self._root

    @property
    def root_positional(self) -> int:
        return np.flatnonzero(self.vertex_index == self.root)[0]

    @property
    def root_location(self) -> np.ndarray:
        return self.vertex_df.loc[self.root, self.spatial_columns].values

    @property
    def parent_node_array(self) -> np.ndarray:
        """Get the parent node array for the skeleton, or -1 for a missing parent."""
        if self._dag_cache.parent_node_array is None:
            self._dag_cache.parent_node_array = gf.build_parent_node_array(
                self.vertices, self.edges_positional
            )
        return self._dag_cache.parent_node_array

    @property
    def branch_points(self) -> np.ndarray:
        "List of branch points of the skeleton based on vertex index"
        return self.vertex_index[self.branch_points_positional]

    @property
    def branch_points_positional(self) -> np.ndarray:
        "List of branch points of the skeleton based on positional index"
        if self._dag_cache.branch_points is None:
            self._dag_cache.branch_points = gf.find_branch_points(self.csgraph_binary)
        return self._dag_cache.branch_points

    @property
    def end_points(self) -> np.ndarray:
        "List of end points of the skeleton based on vertex index"
        return self.vertex_index[self.end_points_positional]

    @property
    def end_points_positional(self) -> np.ndarray:
        "List of end points of the skeleton based on positional index"
        if self._dag_cache.end_points is None:
            self._dag_cache.end_points = gf.find_end_points(self.csgraph_binary)
        return self._dag_cache.end_points

    @property
    def end_points_undirected(self) -> np.ndarray:
        "List of end points of the skeleton based on vertex index potentially including root if a leaf node"
        return self.vertex_index[self.end_points_undirected_positional]

    @property
    def end_points_undirected_positional(self) -> np.ndarray:
        "List of end points of the skeleton based on positional index potentially including root if a leaf node"
        return np.flatnonzero(
            self.csgraph_binary_undirected.sum(axis=1) == 1
        )  # Only one neighbor

    @property
    def branch_points_undirected(self) -> np.ndarray:
        "List of end points of the skeleton based on vertex index potentially including root if a leaf node"
        return self.vertex_index[self.branch_points_undirected_positional]

    @property
    def branch_points_undirected_positional(self) -> np.ndarray:
        "List of end points of the skeleton based on positional index potentially including root if a leaf node"
        return np.flatnonzero(
            self.csgraph_binary_undirected.sum(axis=1) > 2
        )  # More than 2 neighbors

    @property
    def n_end_points(self) -> int:
        "Number of end points in the skeleton"
        return self.end_points.shape[0]

    @property
    def n_branch_points(self) -> int:
        "Number of branch points in the skeleton"
        return self.branch_points.shape[0]

    @property
    def topo_points(self) -> np.ndarray:
        "All vertices not along a segment: branch points, end points, and root node"
        return self.vertex_index[
            np.unique(
                np.concatenate((self.end_points, self.branch_points, [self.root]))
            )
        ]

    @property
    def topo_points_positional(self) -> np.ndarray:
        "All vertices not along a segment: branch points, end points, and root node"
        if self._branch_points is None:
            self._branch_points = gf.find_branch_points(self.csgraph_binary)
        return self._branch_points

    @property
    def n_topo_points(self) -> int:
        "Number of topological points in the skeleton"
        return self.topo_points.shape[0]

    @property
    def verts_edges(self):
        """
        Get the vertices and (positional) edges of the graph as a tuple, which is a common input to many functions.
        """
        return self.vertices, self.edges_positional

    def _set_base_properties(self, base_properties=None):
        if not base_properties:
            self._base_properties["base_root"] = self.root
            self._base_properties["base_vertex_index"] = self.vertex_index
            self._base_properties["base_parent_array"] = self.parent_node_array
            self._base_properties["base_csgraph"] = self.csgraph
            self._base_properties["base_csgraph_binary"] = self.csgraph_binary
        else:
            self._base_properties = copy.deepcopy(base_properties)

    @property
    def base_root(self) -> int:
        return self._base_properties["base_root"]

    @property
    def base_csgraph(self) -> sparse.csr_matrix:
        return self._base_properties["base_csgraph"]

    @property
    def base_csgraph_binary(self) -> sparse.csr_matrix:
        return self._base_properties["base_csgraph_binary"]

    @property
    def base_vertex_index(self) -> Union[str, np.ndarray]:
        return self._base_properties["base_vertex_index"]

    @property
    def base_parent_array(self) -> np.ndarray:
        return self._base_properties["base_parent_array"]

    def _reset_derived_properties(self):
        super()._reset_derived_properties()
        self._dag_cache = gf.DAGCache()

    def _infer_root(self, root: int):
        if root is not None:
            return int(root)
        else:
            potential_roots = np.flatnonzero(self.csgraph_binary.sum(axis=1) == 0)
            if len(potential_roots) == 1:
                return int(potential_roots[0])
            else:
                raise ValueError(
                    "No root specified and edges are not consistent with a single root. Please set a valid root."
                )

    def _apply_root_to_edges(self, root: int, apply_to_all_components: bool = False):
        """Reorient edges so that children are always first in the edge list."""
        if root is None:
            root = self._root

        _, lbls = sparse.csgraph.connected_components(self.csgraph_binary)

        root_comp = lbls[root]
        if apply_to_all_components:
            comps_to_reroot = np.unique(lbls)
        else:
            comps_to_reroot = np.array([root_comp])

        edges_positional_new = self.edges_positional

        for comp in comps_to_reroot:
            if comp == root_comp:
                comp_root = int(root)
            else:
                comp_root = utils.find_far_points_graph(
                    self.csgraph_binary,
                    start_ind=np.flatnonzero(lbls == comp)[0],
                    multicomponent=True,
                )[0]

            d = sparse.csgraph.dijkstra(
                self.csgraph_binary, directed=False, indices=comp_root
            )

            # Make edges in edge list orient as [child, parent]
            # Where each child only has one parent
            # And the root has no parent. (Thus parent is closer than child)
            edge_slice = np.any(
                np.isin(edges_positional_new, np.flatnonzero(lbls == comp)), axis=1
            )

            edge_subset = edges_positional_new[edge_slice]
            is_ordered = d[edge_subset[:, 1]] < d[edge_subset[:, 0]]
            e1 = np.where(is_ordered, edge_subset[:, 0], edge_subset[:, 1])
            e2 = np.where(is_ordered, edge_subset[:, 1], edge_subset[:, 0])
            edges_positional_new[edge_slice] = np.stack((e1, e2)).T

        # Update facets/edges
        for ii in [0, 1]:
            self._morphsync._layers[self.layer_name].facets[ii] = self.vertex_index[
                edges_positional_new[:, ii]
            ]

    def reroot(self, new_root: int, positional=False) -> Self:
        """Reroot to a new index. Important: that this will reset any inherited properties from an unmasked skeleton!

        Parameters
        ----------
        new_root : int
            The new root index to set.
        positional: bool, optional
            Whether the new root is a positional index. If False, the new root is treated as a vertex label.

        Returns
        -------
        Self
        """
        self._reset_derived_properties()
        if not positional:
            new_root = np.flatnonzero(self.vertex_index == new_root)[0]
        self._root = new_root
        self._dag_cache.root = self._root
        self._apply_root_to_edges(new_root)
        self._set_base_properties(
            base_properties={
                "base_root": new_root,
                "base_vertex_index": self.vertex_index,
                "base_parent_array": self.parent_node_array,
                "base_csgraph": self.csgraph,
            }
        )
        return self

    def apply_mask(
        self,
        mask: Optional[np.ndarray] = None,
        new_morphsync: Optional[MorphSync] = None,
    ):
        if new_morphsync is None:
            new_morphsync = self._mask_morphsync(mask=mask)
        return self.__class__(
            name=self.name,
            vertices=new_morphsync.layers.loc[self.layer_name].layer.vertices_df,
            edges=new_morphsync.layers.loc[self.layer_name].layer.edges_df,
            spatial_columns=self.spatial_columns,
            labels=new_morphsync.layers.loc[self.layer_name].layer.nodes,
            morphsync=new_morphsync,
        )

    def distance_to_root(
        self, vertices: Optional[np.ndarray] = None, positional=False
    ) -> np.ndarray:
        """
        Get the distance to the root for each vertex in the skeleton, or for a subset of vertices.

        Parameters
        ----------
        vertices : Optional[np.ndarray]
            The vertices to get the distance from the root for. If None, all vertices are used.
        positional : bool
            If True, the vertices are treated as positional indices. If False, they are treated as vertex labels.

        Returns
        -------
        np.ndarray
            The distance from the root for each vertex.
        """
        # Vertices must be positional for the dijkstra
        vertices, positional = self._vertices_to_positional(vertices, positional)
        if self._dag_cache.distance_to_root is not None:
            dtr = self._dag_cache.distance_to_root
        else:
            dtr = sparse.csgraph.dijkstra(
                self.base_csgraph,
                directed=False,
                indices=self.base_root,
            )
            self._dag_cache.distance_to_root = dtr
        return dtr[vertices]

    def hops_to_root(
        self,
        vertices: Optional[np.ndarray] = None,
        positional=False,
    ) -> np.ndarray:
        """Distance to root in number of hops between vertices. Always works on the base graph, whether the root is masked out or not.

        Parameters
        ----------
        vertices : Optional[np.ndarray]
            The vertices to get the distance from the root for. If None, all vertices are used.
        positional : bool
            If True, the vertices are treated as positional indices. If False, they are treated as vertex labels.

        Returns
        -------
        np.ndarray
            The distance from the root for each vertex.
        """
        vertices, _ = self._vertices_to_positional(vertices, positional)
        if self._dag_cache.hops_to_root is not None:
            htr = self._dag_cache.hops_to_root
        else:
            htr = sparse.csgraph.dijkstra(
                self.base_csgraph_binary,
                directed=False,
                indices=self.base_root,
            )
            self._dag_cache.hops_to_root = htr
        return htr[vertices]

    def child_nodes(self, vertices=None, positional=False) -> dict:
        """Get mapping from vertices to their child nodes.

        Parameters
        ----------
        vertices : Union[np.ndarray, List[int]]
            The vertices to get the child nodes for.
        positional : bool, optional
            Whether the vertices are positional indices. If False, they are treated as vertex labels.

        Returns
        -------
        dict
            A dictionary mapping each vertex to its child nodes.
        """
        vertices, positional = self._vertices_to_positional(vertices, positional)
        cinds = gf.build_child_node_dictionary(vertices, self.csgraph_binary)
        if positional:
            return cinds
        else:
            new_cinds = {}
            for k, v in cinds.items():
                new_cinds[self.vertex_index[k]] = self.vertex_index[v]
            return new_cinds

    def downstream_vertices(
        self, vertex, inclusive=False, positional=False
    ) -> np.ndarray:
        """Get all vertices downstream of a specified vertex

        Parameters
        ----------
        vertex : Union[int, np.ndarray]
            The vertex to get the downstream vertices for.
        inclusive: bool, optional
            Whether to include the specified vertex in the downstream vertices.
        positional : bool, optional
            Whether the vertex is a positional index. If False, it is treated as a vertex label.

        Returns
        -------
        np.ndarray
            The downstream vertices, following the same mode as the positional parameter.
        """
        vertex, positional = self._vertices_to_positional([vertex], positional)
        ds_inds = gf.get_subtree_nodes(
            subtree_root=vertex[0], edges=self.edges_positional
        )
        if inclusive:
            ds_inds = np.concatenate(([vertex[0]], ds_inds))
        if positional:
            return ds_inds
        else:
            return self.vertex_index[ds_inds]

    def cable_length(
        self, vertices: Optional[Union[list, np.ndarray]] = None, positional=False
    ) -> float:
        """The net cable length of the subgraph formed by given vertices. If no vertices are provided, the entire graph is used.

        Parameters
        ----------
        vertices : Optional[Union[list, np.ndarray]]
            The vertices to include in the subgraph. If None, the entire graph is used.
        positional : bool, optional
            Whether the vertices are positional indices. If False, they are treated as vertex labels.

        Returns
        -------
        float
            The net cable length of the subgraph.
        """

        vertices, _ = self._vertices_to_positional(vertices, positional)
        return float(self.csgraph[:, vertices][vertices].sum())

    def lowest_common_ancestor(self, u: int, v: int, positional=False) -> Optional[int]:
        """Get the lowest common ancestor of two vertices in the skeleton.

        Parameters
        ----------
        u : int
            The first vertex.
        v : int
            The second vertex.
        positional : bool, optional
            Whether the vertices are positional indices. If False, they are treated as vertex labels.

        Returns
        -------
        Optional[int]
            The lowest common ancestor of the two vertices, or None if not found.
        """
        uv, positional = self._vertices_to_positional([u, v], positional)
        u = uv[0]
        v = uv[1]
        return gf.lca(
            u,
            v,
            self.vertices,
            self.edges_positional,
            self._dag_cache,
        )

    @property
    def segments_positional(self):
        if self._dag_cache.segments is None:
            self._dag_cache.segments, self._dag_cache.segment_map = gf.build_segments(
                self.vertices,
                self.edges_positional,
                self.branch_points_positional,
                self.child_nodes(positional=True),
                self.hops_to_root(positional=True),
            )
        return self._dag_cache.segments

    @property
    def segments(self):
        if self._dag_cache.segments is None:
            self._dag_cache.segments, self._dag_cache.segment_map = gf.build_segments(
                self.vertices,
                self.edges_positional,
                self.branch_points_positional,
                self.child_nodes(positional=True),
                self.hops_to_root(positional=True),
            )
        return [self.vertices[seg] for seg in self._dag_cache.segments]

    @property
    def segments_plus_positional(self):
        """Segments plus their parent node"""
        segs = self.segments_positional
        return [
            np.concatenate((seg, [self.parent_node_array[seg[-1]]])) for seg in segs
        ]

    @property
    def segments_plus(self):
        """Segments plus their parent node"""
        return [self.vertices[seg] for seg in self.segments_plus_positional]

    @property
    def segment_map(self) -> np.ndarray:
        """Get the mapping from each vertex to its segment index"""
        if self._dag_cache.segments is None:
            self._dag_cache.segments, self._dag_cache.segment_map = gf.build_segments(
                self.vertices,
                self.edges_positional,
                self.branch_points_positional,
                self.child_nodes(positional=True),
                self.hops_to_root(positional=True),
            )
        return self._dag_cache.segment_map

    def expand_to_segment(self, vertices, positional=False):
        """For each vertex in vertices, get the corresponding segment."""
        vertices, positional = self._vertices_to_positional(vertices, positional)
        segment_ids = self.segment_map[vertices]

        if positional:
            return [self.segments_positional[ii] for ii in segment_ids]
        else:
            return [self.segments[ii] for ii in segment_ids]

    @property
    def cover_paths(self):
        """List of 1-d paths in topological order, such that all vertices are covered and a"""

    def __repr__(self) -> str:
        return f"SkeletonSync(name={self.name}, vertices={self.vertices.shape[0]}, edges={self.edges.shape[0]})"


class PointCloudSync(PointMixin):
    def __init__(
        self,
        name: str,
        vertices: Union[np.ndarray, pd.DataFrame],
        spatial_columns: Optional[list] = None,
        *,
        vertex_index: Optional[Union[str, np.ndarray]] = None,
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        morphsync: MorphSync = None,
        linkage: Optional[dict] = None,
    ):
        vertices, spatial_columns, labels = self._setup_properties(
            name=name,
            morphsync=morphsync,
            vertices=vertices,
            spatial_columns=spatial_columns,
            labels=labels,
            vertex_index=vertex_index,
        )
        self._morphsync.add_points(
            points=vertices,
            name=self._name,
            spatial_columns=spatial_columns,
        )
        self._setup_linkage(linkage)
        self._mws = None

    def apply_mask(
        self,
        mask: Optional[np.ndarray] = None,
        new_morphsync: Optional[MorphSync] = None,
    ):
        if new_morphsync is None:
            new_morphsync = self._mask_morphsync(mask=mask)
        return self.__class__(
            name=self.name,
            vertices=new_morphsync.layers.loc[self.layer_name].layer.vertices_df,
            spatial_columns=self.spatial_columns,
            labels=new_morphsync.layers.loc[self.layer_name].layer.nodes,
            morphsync=new_morphsync,
        )

    @property
    def layer_name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"PointCloudSync(name={self.name}, vertices={self.vertices.shape[0]})"

    def distance_to_root(
        self, vertices: Optional[np.ndarray] = None, positional: bool = False
    ) -> np.ndarray:
        """
        Get the distance to the root for each vertex in the point cloud along the skeleton, or for a subset of vertices.

        Parameters
        ----------
        vertices : Optional[np.ndarray]
            The vertices to get the distance to the root for. If None, all vertices are used.
        positional : bool, optional
            If True, the vertices are treated as positional indices. If False, they are treated as vertex labels.
            By default False.

        Returns
        -------
        np.ndarray
            The distance to the root for each vertex.
        """
        if self._mws is None:
            raise ValueError("PointCloud is not attached to a MeshWork object.")
        if self._mws.skeleton is None:
            raise ValueError("Meshwork does not have a Skeleton object.")

        if vertices is None:
            vertices = self.vertex_index
        skel_idx = self.map_index_to_layer(
            layer=SKEL_LAYER_NAME, source_index=vertices, positional=positional
        )
        return self._mws.skeleton.distance_to_root(
            vertices=skel_idx, positional=positional
        )

    def distance_between(
        self,
        vertices: Optional[np.ndarray] = None,
        positional: bool = False,
        via: Literal["skeleton", "graph", "mesh"] = "skeleton",
        limit: Optional[float] = None,
    ) -> np.ndarray:
        """
        Get the distance between each pair of vertices in the point cloud along the skeleton.

        Parameters
        ----------
        vertices : Optional[np.ndarray]
            The vertices to get the distance between. If None, all vertices are used.
        positional : bool, optional
            If True, the vertices are treated as positional indices. If False, they are treated as vertex labels.
            By default False.
        via: Literal["skeleton", "graph", "mesh"], optional
            The method to use for calculating distances. Can be "skeleton", "graph", or "mesh". Default is "skeleton".
        limit: Optional[float], optional
            The maximum distance to consider when calculating distances. If None, no limit is applied.

        Returns
        -------
        np.ndarray
            The distance between each pair of vertices.
        """
        if self._mws is None:
            raise ValueError("PointCloud is not attached to a MeshWork object.")
        if via not in self._morphsync._layers:
            raise ValueError(f"Meshwork does not have a {via.capitalize()} object.")

        vertices, positional = self._vertices_to_positional(vertices, positional)

        target_idx = self.map_index_to_layer(
            layer=via, source_index=vertices, positional=positional
        )
        return self._mws.layers[via].distance_between(
            sources=target_idx, targets=target_idx, positional=positional, limit=limit
        )
