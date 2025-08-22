import copy
import numpy as np
import pandas as pd
from .sync_classes import *
from .data_layers import (
    SKEL_LAYER_NAME,
    GRAPH_LAYER_NAME,
    MESH_LAYER_NAME,
    GraphSync,
    SkeletonSync,
    PointCloudSync,
    PointSyncWork,
)
from typing import Any, Optional, Self, Union
from . import utils


class LayerManager:
    def __init__(
        self,
        managed_layers: dict,
    ):
        self._managed_layers = managed_layers

    @property
    def names(self) -> list:
        """Return a list of managed layer names."""
        return list(self._managed_layers.keys())

    def get(self, name: str) -> Optional[Union[SkeletonSync, GraphSync]]:
        return self._managed_layers.get(name)

    def __getitem__(self, name: str) -> Optional[Union[SkeletonSync, GraphSync]]:
        return self._managed_layers.get(name)

    def __getattr__(self, name: str) -> Optional[Union[SkeletonSync, GraphSync]]:
        if name in self._managed_layers:
            return self._managed_layers[name]
        else:
            raise AttributeError(f'Layer "{name}" does not exist.')

    def __dir__(self):
        return super().__dir__() + list(self._managed_layers.keys())

    def __repr__(self) -> str:
        return str(self.names)


class AnnotationManager:
    def __init__(
        self,
        morphsync: MorphSync,
        managed_layers: list,
        annotation_layers: Optional[list] = None,
    ):
        self._annotations = {}
        self._morphsync = morphsync
        self._managed_layers = managed_layers
        if annotation_layers is not None:
            for layer in annotation_layers:
                if issubclass(type(layer), PointSyncWork):
                    self.add(layer)
                else:
                    raise ValueError("Annotation layers must be point clouds.")

    def add(self, layer: PointCloudSync) -> None:
        self._annotations[layer.name] = layer

    def get(self, name: str, default: Any = None) -> PointCloudSync:
        return self._annotations.get(name, default)

    def __getitem__(self, name: str) -> PointCloudSync:
        return self._annotations[name]

    def __getattr__(self, name: str) -> PointCloudSync:
        if name in self._annotations:
            return self._annotations[name]
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
    SKEL_LN = SKEL_LAYER_NAME
    GRAPH_LN = GRAPH_LAYER_NAME
    MESH_LN = MESH_LAYER_NAME

    def __init__(
        self,
        name: Optional[Union[int, str]] = None,
        morphsync: Optional[MorphSync] = None,
        meta: Optional[dict] = None,
        annotation_layers: Optional[list] = None,
    ):
        if morphsync is None:
            self._morphsync = MorphSync()
        else:
            self._morphsync = copy.deepcopy(morphsync)
            # TODO populate below from this info
        self._name = name
        self._labels = {}
        if meta is None:
            self._meta = dict()
        else:
            self._meta = copy.copy(meta)
        self._managed_layers = {}
        self._layers = LayerManager(self._managed_layers)
        self._annotations = AnnotationManager(
            self._morphsync,
            managed_layers=self._managed_layers,
            annotation_layers=annotation_layers,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def meta(self) -> dict:
        return self._meta

    @property
    def layers(self) -> dict:
        return self._layers

    @property
    def layer_df(self) -> pd.DataFrame:
        return self._morphsync.layers

    def _get_layer(self, layer_name: str):
        return self._managed_layers.get(layer_name)

    @property
    def skeleton(self) -> SkeletonSync:
        if self.SKEL_LN not in self._managed_layers:
            return None
        return self._managed_layers[self.SKEL_LN]

    @property
    def graph(self) -> GraphSync:
        if self.GRAPH_LN not in self._managed_layers:
            return None
        return self._managed_layers[self.GRAPH_LN]

    @property
    def annotations(self) -> AnnotationManager:
        return self._annotations

    def add_skeleton(
        self,
        vertices: Union[np.ndarray, pd.DataFrame, SkeletonSync],
        edges: Union[np.ndarray, pd.DataFrame],
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        *,
        vertex_index: Optional[Union[str, np.ndarray]] = None,
        linkage: Optional[Link] = None,
        spatial_columns: Optional[list] = None,
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
        vertex_index : Optional[Union[str, np.ndarray]]
            The vertex index for the skeleton.
        linkage : Optional[Link]
            The linkage information for the skeleton. Typically, you will define the source vertices for the skeleton if using a graph-to-skeleton mapping.
        spatial_columns: Optional[list] = None
            The spatial columns for the skeleton, if vertices are a dataframe.

        Returns
        -------
        Self
        """
        if self.skeleton is not None:
            raise ValueError('"Skeleton already exists!')

        if isinstance(spatial_columns, str):
            spatial_columns = utils.process_spatial_columns(col_names=spatial_columns)

        if isinstance(vertices, SkeletonSync):
            self._managed_layers[self.SKEL_LN] = SkeletonSync(
                name=self.SKEL_LN,
                vertices=vertices.vertices,
                edges=vertices.edges,
                spatial_columns=vertices.spatial_columns,
                morphsync=self._morphsync,
                linkage=linkage,
            )
        else:
            self._managed_layers[self.SKEL_LN] = SkeletonSync(
                name=self.SKEL_LN,
                vertices=vertices,
                edges=edges,
                labels=labels,
                morphsync=self._morphsync,
                spatial_columns=spatial_columns,
                linkage=linkage,
                vertex_index=vertex_index,
            )
        return self

    def add_graph(
        self,
        vertices: Union[np.ndarray, pd.DataFrame, SkeletonSync],
        edges: Union[np.ndarray, pd.DataFrame],
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        *,
        vertex_index: Optional[Union[str, np.ndarray]] = None,
        spatial_columns: Optional[list] = None,
        linkage: Optional[Link] = None,
    ) -> Self:
        """
        Add the core graph layer to a MeshWork object.
        Additional graph layers can be used, but they must be added separately and with unique names.

        Parameters
        ----------
        vertices : Union[np.ndarray, pd.DataFrame, SkeletonSync]
            The vertices of the graph.
        edges : Union[np.ndarray, pd.DataFrame]
            The edges of the graph.
        labels : Optional[Union[dict, pd.DataFrame]]
            The labels for the graph.
        vertex_index : Optional[Union[str, np.ndarray]]
            The vertex index for the graph.
        spatial_columns: Optional[list] = None
            The spatial columns for the graph, if vertices are a dataframe.

        Returns
        -------
        Self
        """
        if self.graph is not None:
            raise ValueError('"Graph already exists!')
        if isinstance(spatial_columns, str):
            spatial_columns = utils.process_spatial_columns(col_names=spatial_columns)

        if isinstance(vertices, GraphSync):
            self._managed_layers[self.GRAPH_LN] = GraphSync(
                name=self.GRAPH_LN,
                vertices=vertices.vertices,
                edges=vertices.edges,
                spatial_columns=vertices.spatial_columns,
                morphsync=self._morphsync,
                linkage=linkage,
            )
        else:
            self._managed_layers[self.GRAPH_LN] = GraphSync(
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
        """
        Add point annotations to the MeshWork object.

        Parameters
        ----------
        name : str
            The name of the annotation layer.
        vertices : Union[np.ndarray, pd.DataFrame]
            The vertices of the annotation layer.
        spatial_columns : Optional[list]
            The spatial columns for the annotation layer.
        vertex_index : Optional[Union[str, np.ndarray]]
            The vertex index for the annotation layer.
        labels : Optional[Union[dict, pd.DataFrame]]
            The labels for the annotation layer.
        linkage : Optional[Link]
            The linkage information for the annotation layer. Typically, you will define the target vertices for annotations.

        Returns
        -------
        Self
        """

        if name in self._managed_layers:
            raise ValueError(f"Layer '{name}' already exists.")

        if isinstance(spatial_columns, str):
            spatial_columns = utils.process_spatial_columns(col_names=spatial_columns)

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
            repr_list.append(f"graph({self.graph.n_vertices})")
        if self.SKEL_LN in self.layers:
            repr_list.append(f"skel({self.skeleton.n_vertices})")
        if self.MESH_LN in self.layers:
            repr_list.append(f"mesh({self.mesh.n_vertices})")
        return f"MeshWork(name={self.name},{' ' if repr_list else ''}{'+'.join(repr_list)}{',' if repr_list else ''} annotations={self.annotations.names})"
