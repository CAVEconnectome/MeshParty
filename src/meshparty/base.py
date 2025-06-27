import numpy as np
import pandas as pd
import morphsync
from typing import Optional, Union
from . import utils
from abc import ABC, abstractmethod

SKEL_LN = "skeleton"
GRAPH_LN = "graph"
MESH_LN = "mesh"


def _process_vertices(
    vertices: Union[np.ndarray, pd.DataFrame],
    spatial_columns: Optional[list] = None,
    labels: Optional[Union[dict, pd.DataFrame]] = None,
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
    return vertices, spatial_columns, label_columns


# Additional properties for layers with edges
class EdgeMixin(ABC):
    @property
    def edges(self) -> np.ndarray:
        return self.layer.edges

    @property
    def edge_df(self) -> pd.DataFrame:
        return self.layer.edges_df


# General properties for layers with points
class PointMixin(ABC):
    @property
    def name(self) -> str:
        return self._name

    @property
    def layer(self) -> morphsync.Points:
        return self._morphlink.layers.loc[self.layer_name].layer

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

        self._morphlink._layers[self.layer_name].nodes = self.nodes.merge(
            label,
            left_index=True,
            right_index=True,
            how="left",
            validate="1:1",
        )
        self._label_columns += list(label.columns)

    def _mask_morphlink(
        self,
        mask: Optional[np.ndarray] = None,
    ):
        if mask is not None:
            if len(mask) == self.n_vertices:
                mask = mask.astype(bool)
            else:
                mask = self.vertex_index.isin(mask)
        else:
            mask = self.vertex_index

        return self._morphlink.apply_mask(
            layer_name=self.layer_name,
            mask=mask,
        )

    def _process_linkage(self, linkage):
        if linkage is not None:
            if len(linkage) != 1:
                raise ValueError("Mapping must be a dict with one key.")
            target_layer = linkage.keys()[0]
            target_mapping = linkage.values()[0]
            if len(target_mapping) == len(self.vertices):
                self._morphlink.add_link(
                    source=self.layer_name,
                    target=target_layer,
                    mapping=target_mapping,
                )
            else:
                raise ValueError(
                    "Mapping must have the same number of rows as vertices."
                )

    @abstractmethod
    def apply_mask(
        self,
        mask: Optional[np.ndarray] = None,
        new_morphlink: Optional[morphsync.MorphLink] = None,
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
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        morphlink: morphsync.MorphLink = None,
        linkage: Optional[dict] = None,
    ):
        self._name = name
        if morphlink is None:
            self._morphlink = morphsync.MorphLink()
        else:
            self._morphlink = morphlink
        vertices, spatial_columns, label_columns = _process_vertices(
            vertices=vertices,
            spatial_columns=spatial_columns,
            labels=labels,
        )
        self._spatial_columns = spatial_columns
        self._label_columns = label_columns
        self._morphlink.add_graph(
            graph=(vertices, edges),
            name=self.layer_name,
            spatial_columns=spatial_columns,
        )
        self._process_linkage(linkage)

    def apply_mask(
        self,
        mask: Optional[np.ndarray] = None,
        new_morphlink: Optional[morphsync.MorphLink] = None,
    ):
        if new_morphlink is None:
            new_morphlink = self._mask_morphlink(mask=mask)
        return self.__class__(
            name=self.name,
            vertices=new_morphlink.layers.loc[self.layer_name].layer.vertices_df,
            edges=new_morphlink.layers.loc[self.layer_name].layer.edges_df,
            spatial_columns=self.spatial_columns,
            labels=new_morphlink.layers.loc[self.layer_name].layer.nodes,
            morphlink=new_morphlink,
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
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        morphlink: morphsync.MorphLink = None,
        linkage: Optional[dict] = None,
    ):
        self._name = name
        if morphlink is None:
            self._morphlink = morphsync.MorphLink()
        else:
            self._morphlink = morphlink
        vertices, spatial_columns, label_columns = _process_vertices(
            vertices=vertices,
            spatial_columns=spatial_columns,
            labels=labels,
        )
        self._spatial_columns = spatial_columns
        self._label_columns = label_columns
        self._morphlink.add_graph(
            graph=(vertices, edges),
            name=self.layer_name,
            spatial_columns=spatial_columns,
        )
        self._process_linkage(linkage)

    def apply_mask(
        self,
        mask: Optional[np.ndarray] = None,
        new_morphlink: Optional[morphsync.MorphLink] = None,
    ):
        if new_morphlink is None:
            new_morphlink = self._mask_morphlink(mask=mask)
        return self.__class__(
            name=self.name,
            vertices=new_morphlink.layers.loc[self.layer_name].layer.vertices_df,
            edges=new_morphlink.layers.loc[self.layer_name].layer.edges_df,
            spatial_columns=self.spatial_columns,
            labels=new_morphlink.layers.loc[self.layer_name].layer.nodes,
            morphlink=new_morphlink,
        )

    def __repr__(self) -> str:
        return f"SkeletonSync(name={self.name}, vertices={self.vertices.shape[0]}, edges={self.edges.shape[0]})"


class PointCloudSync(PointMixin):
    def __init__(
        self,
        name: str,
        vertices: Union[np.ndarray, pd.DataFrame],
        spatial_columns: Optional[list] = None,
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        morphlink: morphsync.MorphLink = None,
        linkage: Optional[dict] = None,
    ):
        self._name = name

        if morphlink is None:
            self._morphlink = morphsync.MorphLink()

        vertices, spatial_columns, label_columns = _process_vertices(
            vertices=vertices,
            spatial_columns=spatial_columns,
            labels=labels,
        )
        self._spatial_columns = spatial_columns
        self._label_columns = label_columns

        self._morphlink.add_points(
            points=vertices,
            name=self._name,
            spatial_columns=spatial_columns,
        )
        self._process_linkage(linkage)

    def apply_mask(
        self,
        mask: Optional[np.ndarray] = None,
        new_morphlink: Optional[morphsync.MorphLink] = None,
    ):
        if new_morphlink is None:
            new_morphlink = self._mask_morphlink(mask=mask)
        return self.__class__(
            name=self.name,
            vertices=new_morphlink.layers.loc[self.layer_name].layer.vertices_df,
            spatial_columns=self.spatial_columns,
            labels=new_morphlink.layers.loc[self.layer_name].layer.nodes,
            morphlink=new_morphlink,
        )

    @property
    def layer_name(self) -> str:
        return self._name


class MeshWorkSync:
    SKEL_LN = "skeleton"
    GRAPH_LN = "graph"
    MESH_LN = "mesh"

    def __init__(
        self,
        name: Optional[Union[int, str]] = None,
    ):
        self._name = name
        self._morphlink = morphsync.MorphLink()
        self._skeleton = None
        self._graph = None
        self._mesh = None
        self._annotations = None
        self._labels = None

    @property
    def name(self) -> str:
        return self._name

    def add_skeleton(
        self,
        vertices: Union[np.ndarray, pd.DataFrame, SkeletonSync],
        edges: Union[np.ndarray, pd.DataFrame],
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        linkage: Optional[dict] = None,
    ):
        if self.skeleton is not None:
            raise ValueError('"Skeleton already exists!')
        if isinstance(vertices, SkeletonSync):
            self._skeleton = SkeletonSync(
                name=self.name,
                vertices=vertices.vertices,
                edges=vertices.edges,
                spatial_columns=vertices.spatial_columns,
                morphlink=self._morphlink,
                linkage=linkage,
            )
        else:
            self._skeleton = SkeletonSync(
                name=self.name,
                vertices=vertices,
                edges=edges,
                labels=labels,
                morphlink=self._morphlink,
                linkage=linkage,
            )

    @property
    def skeleton(self) -> SkeletonSync:
        if self._skeleton is None:
            return None
        return self._skeleton

    @property
    def graph(self) -> GraphSync:
        if self._graph is None:
            return None
        return self._graph

    def add_graph(
        self,
        vertices: Union[np.ndarray, pd.DataFrame, SkeletonSync],
        edges: Union[np.ndarray, pd.DataFrame],
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        linkage: Optional[dict] = None,
        spatial_columns: Optional[list] = None,
    ):
        if self.graph is not None:
            raise ValueError('"Graph already exists!')

        if isinstance(vertices, GraphSync):
            self._skeleton = GraphSync(
                name=self.name,
                vertices=vertices.vertices,
                edges=vertices.edges,
                spatial_columns=vertices.spatial_columns,
                morphlink=self._morphlink,
                linkage=linkage,
            )
        else:
            self._skeleton = GraphSync(
                name=self.name,
                vertices=vertices,
                edges=edges,
                labels=labels,
                morphlink=self._morphlink,
                linkage=linkage,
            )

    def add_point_annotations(
        self,
        name: str,
        vertices: Union[np.ndarray, pd.DataFrame],
        spatial_columns: Optional[list] = None,
        labels: Optional[Union[dict, pd.DataFrame]] = None,
        linkage: Optional[dict] = None,
    ):
        if isinstance(vertices, PointCloudSync):
            PointCloudSync(
                name=name,
                vertices=vertices.vertices,
                spatial_columns=vertices.spatial_columns,
                morphlink=self._morphlink,
                linkage=linkage,
            )
        else:
            PointCloudSync(
                name=name,
                vertices=vertices,
                spatial_columns=spatial_columns,
                labels=labels,
                morphlink=self._morphlink,
                linkage=linkage,
            )


#     def add_skeleton(
#         self,
#         vertices,
#         edges,

#     ):
#         if isinstance(skeleton, str):
#             skeleton = SkeletonSync(seg_id=int(skeleton))
#         elif not isinstance(skeleton, SkeletonSync):
#             raise ValueError("Skeleton must be a SkeletonSync or str.")

#         if name is None:
#             name = self.SKEL_LN
#         self._morphlink.add_skeleton(
#             skeleton=skeleton,
#             name=name,
#     )
