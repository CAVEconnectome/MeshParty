# Additional properties for layers with edges
from abc import ABC, abstractmethod
from typing import Optional, Union

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
        return {v: ii for ii, v in enumerate(self.vertex_index)}

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


class SkeletonSync(PointMixin, EdgeMixin):
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

        # self._root = self._infer_root(root)
        # self._parent_node_array = self._apply_root_to_edges(root)

        # if inherited_properties is None:
        #     self._set_base_properties(
        #         base_properties={
        #             "base_root": self.root,
        #             "base_vertex_index": self.vertex_index,
        #             "base_parent_array": self.parent_node_array,
        #             "base_csgraph": self.csgraph,
        #         }
        #     )
        # else:
        #     self._set_base_properties(base_properties=inherited_properties)

    @property
    def root(self) -> int:
        return self._root

    @property
    def parent_node_array(self) -> np.ndarray:
        return self._parent_node_array

    def _set_base_properties(self, base_properties=None):
        if not base_properties:
            self._base_properties["base_root"] = self.root  # positional index
            self._base_properties["base_vertex_index"] = (
                self.vertex_index
            )  # vertex indices
            self._base_properties["base_parent_array"] = (
                self._parent_node_array
            )  # positional indices into vertex_index
            self._base_properties["base_csgraph"] = (
                self.csgraph
            )  # positional into base_vertex_index
        else:
            self._base_properties = copy.deepcopy(base_properties)

    @property
    def base_properties(self) -> dict:
        "Key graph properties of the original skeleton."
        return self._base_properties

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
        if root is None:
            root = self._root

        _, lbls = sparse.csgraph.connected_components(self.csgraph_binary)

        root_comp = lbls[root]
        if apply_to_all_components:
            comps_to_reroot = np.unique(lbls)
        else:
            comps_to_reroot = np.array([root_comp])

        new_parent_node_array = np.full(self.n_vertices, -1, dtype=int)
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
            is_ordered = d[edge_subset[:, 0]] > d[edge_subset[:, 1]]
            e1 = np.where(is_ordered, edge_subset[:, 0], edge_subset[:, 1])
            e2 = np.where(is_ordered, edge_subset[:, 1], edge_subset[:, 0])
            edges_positional_new[edge_slice] = np.stack((e1, e2)).T
            new_parent_node_array[e1] = e2

        # Update facets/edges
        for ii in [0, 1]:
            self._morphsync._layers[self.layer_name].facets[ii] = self.vertex_index[
                edges_positional_new[:, ii]
            ]
        return new_parent_node_array

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
