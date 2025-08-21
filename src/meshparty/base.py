import copy
import numpy as np
import pandas as pd
import morphsync as sync
import fastremap
from typing import Any, Optional, Self, Union
from . import utils
from abc import ABC, abstractmethod
import dataclasses

SKEL_LN = "skeleton"
GRAPH_LN = "graph"
MESH_LN = "mesh"


@dataclasses.dataclass
class Link:
    """
    Represents the linkage mapping information.

    Parameters
    ----------
    mapping: Union[list[int], str]
        The mapping information between the source and target layers.
        If a string, will be a column name in the target's vertex dataframe
    source: str
        The name of the source layer, typically the one with more vertices than the target. E.g. a graph or mesh to a skeleton, or a skeleton to point annotations.
    target: str
        The name of the target layer, typically the one with fewer vertices. E.g. a skeleton from a graph or mesh, or point annotations from a skeleton.
    map_value_is_index: bool, optional
        If True, assumes the values in the list or the mapping are a non-positional dataframe index
    """

    mapping: Union[list[int], str]
    source: Optional[str] = None
    target: Optional[str] = None
    map_value_is_index: bool = True

    def __post_init__(self):
        if isinstance(self.mapping, (list, np.ndarray)):
            self.mapping = np.array(self.mapping, dtype=int)

    def mapping_to_index(self, vertex_data: pd.DataFrame):
        if self.map_value_is_index:
            return self.mapping
        else:
            return vertex_data.index.values[self.mapping]


def _process_vertices(
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


# Additional properties for layers with edges
class EdgeMixin(ABC):
    @property
    def edges(self) -> np.ndarray:
        return self.layer.edges

    @property
    def edge_df(self) -> pd.DataFrame:
        return self.layer.edges_df

    def _map_edges_to_index(self, edges, vertex_indices):
        index_map = {ii: v for ii, v in enumerate(vertex_indices)}
        return fastremap.remap(edges, index_map)


# General properties for layers with points
class PointMixin(ABC):
    def _setup_properties(
        self,
        name: str,
        morphsync: Optional[sync.MorphSync] = None,
        vertices: Union[np.ndarray, pd.DataFrame] = None,
        spatial_columns: Optional[list] = None,
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        vertex_index: Optional[Union[str, np.ndarray]] = None,
    ):
        self._name = name
        if morphsync is None:
            self._morphsync = sync.MorphSync()
        else:
            self._morphsync = morphsync
        vertices, spatial_columns, label_columns = _process_vertices(
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
    def layer(self) -> sync.Points:
        return self._get_layer(self.layer_name)

    def _get_layer(self, layer_name: str) -> sync.base.FacetFrame:
        return self._morphsync._layers[layer_name]

    @property
    def vertices(self) -> np.ndarray:
        return self.layer.vertices

    @property
    def vertex_df(self) -> pd.DataFrame:
        return self.layer.vertices_df

    @property
    def vertex_index(self) -> pd.Index:
        return self.layer.vertices_index

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
        new_morphsync: Optional[sync.MorphSync] = None,
    ):
        pass


class GraphSync(PointMixin, EdgeMixin):
    layer_name = GRAPH_LN

    def __init__(
        self,
        name: str,
        vertices: Union[np.ndarray, pd.DataFrame],
        edges: Union[np.ndarray, pd.DataFrame],
        spatial_columns: Optional[list] = None,
        *,
        vertex_index: Optional[Union[str, np.ndarray]] = None,
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        morphsync: sync.MorphSync = None,
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
        new_morphsync: Optional[sync.MorphSync] = None,
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
    layer_name = SKEL_LN

    def __init__(
        self,
        name: str,
        vertices: Union[np.ndarray, pd.DataFrame],
        edges: Union[np.ndarray, pd.DataFrame],
        spatial_columns: Optional[list] = None,
        *,
        vertex_index: Optional[Union[str, np.ndarray]] = None,
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        morphsync: sync.MorphSync = None,
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
        new_morphsync: Optional[sync.MorphSync] = None,
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
        morphsync: sync.MorphSync = None,
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
        new_morphsync: Optional[sync.MorphSync] = None,
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


class AnnotationManager:
    def __init__(
        self,
        morphsync: sync.MorphSync,
        annotation_layers: Optional[list] = None,
    ):
        self._annotations = {}
        self._morphsync = morphsync
        if annotation_layers is not None:
            for layer in annotation_layers:
                if issubclass(type(layer), PointCloudSync):
                    self.add(layer)
                else:
                    raise ValueError(
                        "Annotation layers must be instances of PointCloudSync."
                    )

    def add(self, layer: PointCloudSync) -> None:
        self._annotations[layer.name] = layer

    def get(self, name: str, default: Any = None) -> PointCloudSync:
        if name in self._annotations:
            return getattr(self._morphsync, name)
        else:
            return default

    def __getattr__(self, name: str) -> PointCloudSync:
        if name in self._annotations:
            return self._morphsync.layers.loc[name].layer
        else:
            raise AttributeError(f'Annotation "{name}" does not exist.')

    def __dir__(self):
        return super().__dir__() + list(self._annotations.keys())

    @property
    def names(self) -> list:
        """Return a list of annotation names."""
        return list(self._annotations.keys())

    def __contains__(self, name: str) -> bool:
        """Check if an annotation exists by name."""
        return name in self._annotations

    def __len__(self) -> int:
        """Return the number of annotations."""
        return len(self._annotations)

    def __repr__(self) -> str:
        return f"AnnotationManager(annotations={list(self._annotations.keys())})"


class MeshWorkSync:
    SKEL_LN = "skeleton"
    GRAPH_LN = "graph"
    MESH_LN = "mesh"

    def __init__(
        self,
        name: Optional[Union[int, str]] = None,
        morphsync: Optional[sync.MorphSync] = None,
        meta: Optional[dict] = None,
        annotation_layers: Optional[list] = None,
    ):
        if morphsync is None:
            self._morphsync = sync.MorphSync()
        else:
            self._morphsync = copy.deepcopy(morphsync)
        self._name = name
        self._annotations = AnnotationManager(
            self._morphsync, annotation_layers=annotation_layers
        )
        self._labels = None  # todo: populate
        if meta is None:
            self._meta = dict()
        else:
            self._meta = copy.copy(meta)

    @property
    def name(self) -> str:
        return self._name

    @property
    def meta(self) -> dict:
        return self._meta

    @property
    def layers(self) -> dict:
        return self._morphsync._layers

    @property
    def layer_df(self) -> pd.DataFrame:
        return self._morphsync.layers

    def add_skeleton(
        self,
        vertices: Union[np.ndarray, pd.DataFrame, SkeletonSync],
        edges: Union[np.ndarray, pd.DataFrame],
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        linkage: Optional[Link] = None,
        vertex_index: Optional[Union[str, np.ndarray]] = None,
    ) -> Self:
        """
        Add a skeleton layer to the MorphSync.

        Parameters
        ----------
        vertices : Union[np.ndarray, pd.DataFrame, SkeletonSync]
            The vertices of the skeleton.
        edges : Union[np.ndarray, pd.DataFrame]
            The edges of the skeleton.
        labels : Optional[Union[dict, pd.DataFrame]]
            The labels for the skeleton.
        linkage : Optional[Link]
        """
        if self.skeleton is not None:
            raise ValueError('"Skeleton already exists!')
        if isinstance(vertices, SkeletonSync):
            SkeletonSync(
                name=self.SKEL_LN,
                vertices=vertices.vertices,
                edges=vertices.edges,
                spatial_columns=vertices.spatial_columns,
                morphsync=self._morphsync,
                linkage=linkage,
            )
        else:
            SkeletonSync(
                name=self.SKEL_LN,
                vertices=vertices,
                edges=edges,
                labels=labels,
                morphsync=self._morphsync,
                linkage=linkage,
                vertex_index=vertex_index,
            )
        return self

    @property
    def skeleton(self) -> SkeletonSync:
        if self.SKEL_LN not in self.layers:
            return None
        return self.layers[self.SKEL_LN]

    @property
    def graph(self) -> GraphSync:
        if self.GRAPH_LN not in self.layers:
            return None
        return self.layers[self.GRAPH_LN]

    @property
    def annotations(self) -> AnnotationManager:
        return self._annotations

    def add_graph(
        self,
        vertices: Union[np.ndarray, pd.DataFrame, SkeletonSync],
        edges: Union[np.ndarray, pd.DataFrame],
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        spatial_columns: Optional[list] = None,
        vertex_index: Optional[Union[str, np.ndarray]] = None,
        linkage: Optional[Link] = None,
    ) -> Self:
        if self.graph is not None:
            raise ValueError('"Graph already exists!')

        if isinstance(vertices, GraphSync):
            GraphSync(
                name=self.GRAPH_LN,
                vertices=vertices.vertices,
                edges=vertices.edges,
                spatial_columns=vertices.spatial_columns,
                morphsync=self._morphsync,
                linkage=linkage,
            )
        else:
            GraphSync(
                name=self.GRAPH_LN,
                vertices=vertices,
                edges=edges,
                labels=labels,
                morphsync=self._morphsync,
                spatial_columns=spatial_columns,
                linkage=linkage,
                vertex_index=vertex_index,
            )
        return self

    def add_point_annotations(
        self,
        name: str,
        vertices: Union[np.ndarray, pd.DataFrame],
        spatial_columns: Optional[list] = None,
        *,
        vertex_index: Optional[Union[str, np.ndarray]] = None,
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        linkage: Optional[Link] = None,
    ) -> Self:
        if isinstance(vertices, PointCloudSync):
            anno = PointCloudSync(
                name=name,
                vertices=vertices.vertices,
                spatial_columns=vertices.spatial_columns,
                morphsync=self._morphsync,
                linkage=linkage,
            )
        else:
            anno = PointCloudSync(
                name=name,
                vertices=vertices,
                spatial_columns=spatial_columns,
                vertex_index=vertex_index,
                labels=labels,
                morphsync=self._morphsync,
                linkage=linkage,
            )
        self._annotations.add(anno)
        return self

    def __repr__(self) -> str:
        repr_list = []
        if self.GRAPH_LN in self.layers:
            repr_list.append("graph")
        if self.SKEL_LN in self.layers:
            repr_list.append("skel")
        if self.MESH_LN in self.layers:
            repr_list.append("mesh")
        return f"MeshWork(name={self.name},{' ' if repr_list else ''}{'+'.join(repr_list)}{',' if repr_list else ''} annotations={self.annotations.names})"
