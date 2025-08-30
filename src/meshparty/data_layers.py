# Additional properties for layers with edges
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Tuple, Union, Self, TYPE_CHECKING
import uuid

from scipy import sparse, spatial

from . import utils

import copy
import fastremap
import numpy as np
import pandas as pd
from .sync_classes import (
    MorphSync,
    PointSyncWork,
    GraphSyncWork,
    Facet,
    Link,
)
from . import graph_functions as gf

if TYPE_CHECKING:
    from .base import MeshWorkSync

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
        st, _ = self._vertices_to_positional([source, target], positional)
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
        self._kdtree = None
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
        vertex_index: Optional[np.ndarray] = None,
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
                if vertex_index is None:
                    vertex_index_map = self.vertex_index_map
                else:
                    vertex_index_map = {v: i for i, v in enumerate(vertex_index)}
                vertices = fastremap.remap(vertices, vertex_index_map)
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

    @property
    def kdtree(self) -> spatial.KDTree:
        if self._kdtree is None:
            self._kdtree = spatial.KDTree(self.vertices)
        return self._kdtree

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

    def _map_range_to_range(self, layer: str, source_index: np.ndarray) -> np.ndarray:
        """This takes the dataframe in `get_mapping` and returns all values in the target layer, without respecting a 1:1 mapping between indices"""
        mapping = self._morphsync.get_mapping(
            source=self.layer_name, target=layer, source_index=source_index
        )
        return mapping[layer].values

    def _map_index_one_to_one(
        self, layer: str, source_index: np.ndarray, validate: bool = False
    ) -> np.ndarray:
        """Takes the list of source indices and returns one target index for each source. It promises to maintain the order of the source indices."""
        mapping = self._morphsync.get_mapping(
            source=self.layer_name, target=layer, source_index=source_index
        )
        if validate:
            if np.any(mapping.index.duplicated()):
                raise ValueError(
                    f"Ambiguous index mapping from {self.layer_name} to {layer}."
                )
        return mapping[~mapping.index.duplicated(keep="first")].loc[source_index].values

    def _map_index_to_list_of_lists(self, layer: str, source_index: np.ndarray) -> dict:
        """Takes the list of source indices and returns a list of all target indices for each source."""
        mapping = self._morphsync.get_mapping(
            source=self.layer_name, target=layer, source_index=source_index
        )
        mapping_dict = mapping.groupby(by=mapping.index).agg(list).to_dict()
        return {k: np.array(v) for k, v in mapping_dict.items()}

    def map_labels_to_layer(
        self, labels: Union[str, list], layer: str, agg: Union[str, dict]
    ) -> pd.DataFrame:
        """Map labels from one layer to another.

        Parameters
        ----------
        labels: Union[str, list]
            The labels to map from the source layer.
        layer: str
            The target layer to map the labels to.
        agg: Union[str, dict]
            The aggregation method to use when mapping the labels.
            This can take anything pandas `groupby.agg` takes, as well as
            "majority" which will is a majority vote across the mapped indices
            via the stats.mode function.

        Returns
        -------
        pd.DataFrame
            The mapped labels for the target layer.
        """
        if layer == self.layer_name:
            return self.nodes[labels]
        if isinstance(labels, str):
            source_labels = [labels]
        mapping = self._morphsync.get_mapping(
            source=self.layer_name, target=layer, source_index=self.vertex_index
        )
        mapping_merged = mapping.to_frame().merge(
            self.nodes[labels],
            left_index=True,
            right_index=True,
            how="left",
        )
        if agg == "majority":
            agg = utils.majority_agg()
        return (
            mapping_merged.groupby(layer)
            .agg(agg)
            .loc[self._morphsync._layers[layer].nodes.index]
        )

    def _map_index_to_layer(
        self,
        layer: str,
        source_index: np.ndarray,
        positional: bool,
        how: str,
        validate: bool = False,
    ):
        if layer == self.layer_name:
            return source_index
        if source_index is None:
            source_index = self.vertex_index
        source_index = np.array(source_index)
        if positional or np.issubdtype(source_index.dtype, np.bool):
            source_index = self.vertex_index[source_index]
        if layer in self._morphsync._layers:
            match how:
                case "one_to_one":
                    mapping = self._map_index_one_to_one(
                        layer=layer, source_index=source_index, validate=validate
                    )
                case "range_to_range":
                    mapping = self._map_range_to_range(
                        layer=layer, source_index=source_index
                    )
                case "one_to_list":
                    mapping = self._map_index_to_list_of_lists(
                        layer=layer, source_index=source_index
                    )
            if positional:
                if isinstance(mapping, np.ndarray):
                    mapping = fastremap.remap(
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
                elif isinstance(mapping, dict):
                    mapping = {
                        self.vertex_index_map[k]: fastremap.remap(
                            v,
                            {
                                int(kk): ii
                                for ii, kk in enumerate(
                                    np.array(
                                        self._morphsync._layers[
                                            layer
                                        ].vertices_index.values
                                    )
                                )
                            },
                        )
                        for k, v in mapping.items()
                    }
            return mapping
        else:
            raise ValueError(f"Layer '{layer}' does not exist.")

    def map_index_to_layer(
        self,
        layer: str,
        source_index=None,
        positional: bool = False,
        validate: bool = False,
    ) -> Optional[int]:
        """Map each vertex index from the current layer to a single index in the specified layer.

        Parameters
        ----------
        layer : str
            The target layer to map the index to.
        source_index : Optional[np.ndarray]
            The source index to map from. If None, all vertices are used. Can also be a boolean array.
        positional : bool
            Whether to treat source_index and mapped index as positional (i_th element of the array) or as a dataframe index.
        validate : bool
            Whether to raise an error is the mapping is ambiguous, i.e. it is not clear which target index to use.

        Returns
        -------
        Optional[int]
            The mapped index in the target layer, or None if not found.
            There will be exactly one target index for each source index, no matter how many viable target indices there are.
            If `positional` is True, the mapping is based on the position of the vertices not the dataframe index.
        """
        return self._map_index_to_layer(
            layer=layer,
            source_index=source_index,
            positional=positional,
            how="one_to_one",
            validate=validate,
        )

    def map_region_to_layer(
        self, layer: str, source_index=None, positional: bool = False
    ) -> Optional[int]:
        """Map each vertex index from the current layer to the specified layer.

        Parameters
        ----------
        layer : str
            The target layer to map the index to.
        source_index : Optional[np.ndarray]
            The source indices to map from. If None, all vertices are used. Can also be a boolean array.
        positional : bool
            Whether to treat source_index and mapped index as positional (i_th element of the array) or as a dataframe index.

        Returns
        -------
        Optional[int]
            All mapped indices in the target layer, or None if not found.
            Not necessarily the same length as the source indices, because it maps a region to another region.
            If `positional` is True, the mapping is based on the position of the vertices not the dataframe index.
        """
        return self._map_index_to_layer(
            layer=layer,
            source_index=source_index,
            positional=positional,
            how="range_to_range",
        )

    def map_index_to_layer_region(
        self, layer: str, source_index=None, positional: bool = False
    ) -> dict:
        """Map each vertex index from the current layer to a list of all appropriate vertices in the target layer.

        Parameters
        ----------
        layer : str
            The target layer to map the index to.
        source_index : Optional[np.ndarray]
            The source indices to map from. If None, all vertices are used. Can also be a boolean array.
        positional : bool
            Whether to treat source_index and mapped index as positional (i_th element of the array) or as a dataframe index.

        Returns
        -------
        dict
            A dictionary mapping each source index to a list of all mapped target indices.
        """
        return self._map_index_to_layer(
            layer=layer,
            source_index=source_index,
            positional=positional,
            how="one_to_list",
        )

    def map_mask_to_layer(
        self, layer: str, source_index=None, positional: bool = False
    ) -> Optional[np.ndarray]:
        """Map each vertex index from the current layer to a region (list of indices) in the specified layer.

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
        Optional[np.ndarray]
            The mapped indices in the target layer, or None if not found.
            There may be multiple target indices for each source index, depending on the region mapping.
            If `positional` is True, the mapping is based on the position of the vertices not the dataframe index.
        """
        mapping = self._map_index_to_layer(
            layer=layer,
            source_index=source_index,
            positional=positional,
            how="range_to_range",
        )
        return self._layers[layer].nodes.index.isin(mapping).values

    def _mask_morphsync(self, mask: np.ndarray, positional: bool = False):
        if positional:
            bool_mask = np.full(self.n_vertices, False)
            bool_mask[mask] = True
            mask = bool_mask
        else:
            mask = np.array(mask)
            if len(mask) == self.n_vertices and np.issubdtype(mask.dtype, np.bool_):
                mask = mask.astype(bool)
            else:
                mask = self.vertex_index.isin(mask)
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

    @classmethod
    @abstractmethod
    def _from_existing(cls, new_morphsync: MorphSync, old_obj: Self) -> Self:
        pass

    def apply_mask(
        self,
        mask: np.ndarray,
        positional: bool = False,
        self_only: bool = False,
    ) -> Union[Self, "MeshWorkSync"]:
        """Apply a mask on the current layer. Returns a new object with the masked morphsync.
        If the object is associated with a MeshworkSync, a new MeshworkSync will be created, otherwise
        a new object of the same class will be returned.

        Properties
        ----------
        mask: np.ndarray
            The mask to apply, either in boolean, vertex index, or positional index form.
        positional: bool
            If providing indices, specify if they are positional indices (True) or vertex indices (False).
        self_only: bool
            If True, only apply the mask to the current object and not to any associated MeshWorkSync.

        Returns
        -------
        masked_object Union[Self, "MeshWorkSync"]
            Either a new object of the same class or a new MeshWorkSync will be returned.
        """
        new_morphsync = self._mask_morphsync(mask=mask, positional=positional)
        if self._mws is None or self_only:
            return self.__class__._from_existing(
                new_morphsync=new_morphsync, old_obj=self
            )
        else:
            return self._mws.__class__._from_existing(
                new_morphsync=new_morphsync, old_obj=self._mws
            )

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
        existing: bool = False,
    ):
        vertices, spatial_columns, labels = self._setup_properties(
            name=name,
            morphsync=morphsync,
            vertices=vertices,
            spatial_columns=spatial_columns,
            labels=labels,
            vertex_index=vertex_index,
        )

        self._mws = None

        if not existing:
            if vertex_index:
                edges = self._map_edges_to_index(edges, vertices.index)
            self._morphsync.add_graph(
                graph=(vertices, edges),
                name=self.layer_name,
                spatial_columns=spatial_columns,
            )
            self._setup_linkage(linkage)

    @classmethod
    def _from_existing(
        cls,
        new_morphsync: MorphSync,
        old_obj: Self,
    ) -> Self:
        "Generate a new SkeletonSync derived from an existing morphsync and skeleton metadata, no need for new vertices or edges."
        new_obj = cls(
            name=old_obj.name,
            vertices=old_obj.nodes,
            edges=old_obj.edges,
            spatial_columns=old_obj.spatial_columns,
            morphsync=new_morphsync,
            existing=True,
        )
        return new_obj

    def map_annotations_to_label(
        self,
        annotation: str,
        distance_threshold: float,
        agg="count",
        chunk_size: int = 1000,
        validate: bool = False,
    ):
        """Aggregates a point annotation to a label on the layer"""
        idx_list, prox_list = gf.build_proximity_lists_chunked(
            self.vertices,
            self.csgraph,
            distance_threshold=distance_threshold,
            chunk_size=chunk_size,
        )
        prox_df = pd.DataFrame(
            {
                "idx": self.vertex_index[idx_list],
                "prox_idx": self.vertex_index[prox_list],
            }
        )
        anno_df = self._morphsync._layers[annotation].nodes
        local_vertex = self._mws.annotations[annotation].map_index_to_layer(
            self.layer_name, validate=validate
        )
        local_vertex_col = f"temp_{uuid.uuid4().hex}"
        anno_df[local_vertex_col] = local_vertex
        prox_df = prox_df.merge(
            anno_df,
            left_on="prox_idx",
            right_on=local_vertex_col,
            how="left",
        )
        anno_df.drop(columns=local_vertex_col, inplace=True)
        if agg == "count":
            count_ser = prox_df.groupby("idx")[local_vertex_col].count()
            count_ser.name = f"{annotation}_count"
            return count_ser
        elif isinstance(agg, dict):
            agg_df = prox_df.groupby("idx").agg(**agg)
            return agg_df
        else:
            raise ValueError(
                f"Unknown aggregation type: {agg}. Must be 'count' or a dict."
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

        if inherited_properties is None:
            # Add as a morphsync layer
            if vertex_index:
                edges = self._map_edges_to_index(edges, vertices.index)
            self._morphsync.add_graph(
                graph=(vertices, edges),
                name=self.layer_name,
                spatial_columns=spatial_columns,
            )
            self._setup_linkage(linkage)

            # Establish the root and then build the base properties
            self._root = self._infer_root(root)
            self._dag_cache = gf.DAGCache(
                root=self.root_positional
            )  # Cache of properties associated with rooted skeletons
            self._dag_cache.parent_node_array = self._apply_root_to_edges(root)

            self._set_base_properties(
                base_properties={
                    "base_root": self.root_positional,
                    "base_vertex_index": self.vertex_index,
                    "base_parent_array": self.parent_node_array,
                    "base_csgraph": self.csgraph,
                    "base_csgraph_binary": self.csgraph_binary,
                }
            )
        else:
            # Infer all values from inherited properties and/or the existing morphsync.
            self._set_base_properties(base_properties=inherited_properties)

            old_root_pos = inherited_properties.get("base_root")
            old_root_idx = inherited_properties.get("base_vertex_index")[old_root_pos]

            if old_root_idx in self.vertex_index:
                self._root = old_root_idx
            else:
                self._root = None

            self._dag_cache = gf.DAGCache(root=self.root_positional)
            self._dag_cache.parent_node_array = gf.build_parent_node_array(
                self.vertices, self.edges_positional
            )

        self._mws = None

    @classmethod
    def _from_existing(
        cls,
        new_morphsync: MorphSync,
        old_obj: Self,
    ) -> Self:
        "Generate a new SkeletonSync derived from an existing morphsync and skeleton metadata, no need for new vertices or edges."
        new_obj = cls(
            name=old_obj.name,
            vertices=old_obj.nodes,
            edges=old_obj.edges,
            spatial_columns=old_obj.spatial_columns,
            morphsync=new_morphsync,
            inherited_properties=old_obj._base_properties,
        )
        return new_obj

    @property
    def root(self) -> Optional[int]:
        return self._root

    @property
    def root_positional(self) -> Optional[int]:
        if self._root is None:
            return None
        return np.flatnonzero(self.vertex_index == self.root)[0]

    @property
    def root_location(self) -> Optional[np.ndarray]:
        if self.root == -1:
            return None
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

        return self.vertex_index[self.topo_points_positional]

    @property
    def topo_points_positional(self) -> np.ndarray:
        "All vertices not along a segment: branch points, end points, and root node"
        bp_ep_rp = np.concatenate(
            (self.branch_points_positional, self.end_points_positional)
        )
        if self.root is not None:
            bp_ep_rp = np.concatenate((bp_ep_rp, [self.root_positional]))
        return np.unique(bp_ep_rp)

    @property
    def n_topo_points(self) -> int:
        "Number of topological points in the skeleton"
        return self.topo_points.shape[0]

    @property
    def parent_free_nodes(self) -> np.ndarray:
        "List of nodes by vertex index that do not have any parents, including any root node."
        return self.vertex_index[self.parent_free_nodes_positional]

    @property
    def parent_free_nodes_positional(self) -> np.ndarray:
        "List of nodes by positional index that do not have any parents, including any root node."
        return np.flatnonzero(self.parent_node_array == -1)

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
        self._mws.apply_mask(mask=mask, new_morphsync=new_morphsync)

    def distance_to_root(
        self, vertices: Optional[np.ndarray] = None, positional=False
    ) -> np.ndarray:
        """
        Get the distance to the root for each vertex in the skeleton, or for a subset of vertices.
        Always uses the original skeleton topology, so that root is inherited from the original root even if it is
        currently masked out. E.g. if you mask an axon only, you can still get distance to the root soma even if
        the soma is not in your current object.

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
        vertices, positional = self._vertices_to_positional(
            vertices, positional, vertex_index=self.base_vertex_index
        )
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
    def cover_paths(self):
        return [self.vertex_index[path] for path in self.cover_paths_positional]

    @property
    def cover_paths_positional(self):
        if self._dag_cache.cover_paths is None:
            self._dag_cache.cover_paths = gf.build_cover_paths(
                self.end_points_positional,
                self.parent_node_array,
                self.distance_to_root(positional=True),
                self._dag_cache,
            )
        return self._dag_cache.cover_paths

    def cover_paths_specific(self, sources: Union[np.ndarray, list], positional=False):
        """Get cover paths starting from specific source vertices.

        Parameters
        ----------
        sources : Union[np.ndarray, list]
            The source vertices to start the cover paths from.
        positional : bool, optional
            Whether the sources are positional indices. If False, they are treated as vertex labels.

        Returns
        -------
        list
            A list of cover paths, each path is a list of vertex indices, ordered as the typical `cover_paths` method.
        """
        sources, positional = self._vertices_to_positional(sources, positional)
        cps = gf.compute_cover_paths(
            sources,
            self.parent_node_array,
            self.distance_to_root(positional=True),
        )
        if positional:
            return [self.vertex_index[path] for path in cps]
        return cps

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
    def half_edge_length(self):
        """
        Get the sum length of half-edges from a vertices to all parents and children.
        """
        return np.array(self.csgraph_undirected.sum(axis=0)).flatten() / 2

    def map_annotations_to_label(
        self,
        annotation: str,
        distance_threshold: float,
        agg: Union[Literal["count", "density"], dict] = "count",
        chunk_size: int = 1000,
        validate: bool = False,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Aggregates a point annotation to a label on the layer based on a maximum proximity.

        Parameters
        ----------
        annotation : str
            The name of the annotation to project.
        distance_threshold : float
            The maximum distance to consider for projecting annotations.
        agg : Union[Literal["count", "density"], dict], optional
            The aggregation method to use. Can be "count", or a dict specifying custom aggregations
            on the annotations properties as per the `groupby.agg` method.
            * "count" returns how many annotations are within the given radius.
            * "density" returns the count of annotations divided by the subgraph path length measured in half-edge-lengths per vertex.
            * To make a new label called "aggregate_label" that is the median "size" of a point annotation,
            it would be {"aggregate_label": ('size', 'median')}. Multiple labels can be specified at the same time in this manner.
        chunk_size : int, optional
            The size of chunks to process at a time, which limits memory consumption. Defaults to 1000.

        Returns
        -------
        pd.Series or pd.DataFrame
            A series (with 'count' or 'density') or dataframe (with dictionary agg) containing the projected annotation values for each vertex.
        """
        if agg == "density":
            agg_temp = "count"
        else:
            agg_temp = agg
        result = super().map_annotations_to_label(
            annotation,
            distance_threshold,
            agg=agg_temp,
            chunk_size=chunk_size,
            validate=validate,
        )
        if agg == "density":
            count_len_df = pd.concat(
                (
                    pd.Series(
                        data=self.half_edge_length,
                        index=self.vertex_index,
                        name="net_length",
                    ),
                    result,
                ),
                axis=1,
            )
            return pd.Series(
                data=count_len_df[result.name] / count_len_df["net_length"],
                index=count_len_df.index,
                name=f"{annotation}_density",
            )
        else:
            return result

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
        existing: bool = False,
    ):
        vertices, spatial_columns, labels = self._setup_properties(
            name=name,
            morphsync=morphsync,
            vertices=vertices,
            spatial_columns=spatial_columns,
            labels=labels,
            vertex_index=vertex_index,
        )
        if not existing:
            self._morphsync.add_points(
                points=vertices,
                name=self._name,
                spatial_columns=spatial_columns,
            )
            self._setup_linkage(linkage)
        self._mws = None

    @classmethod
    def _from_existing(cls, new_morphsync, old_obj) -> Self:
        return cls(
            name=old_obj.name,
            vertices=old_obj.nodes,
            spatial_columns=old_obj.spatial_columns,
            morphsync=new_morphsync,
            existing=True,
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
