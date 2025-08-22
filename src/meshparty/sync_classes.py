import dataclasses
from typing import Optional, Union
import morphsync as sync
import numpy as np
import pandas as pd

__all__ = [
    "MorphSync",
    "GraphSyncWork",
    "MeshSyncWork",
    "SkeletonSyncWork",
    "PointSyncWork",
    "Facet",
    "is_syncwork_object",
    "Link",
]

Facet = sync.base.FacetFrame  # Alias for FacetFrame
MorphSync = sync.MorphSync


def is_syncwork_object(obj):
    """
    Check if the object is an instance of a sync work class.
    """
    return isinstance(
        obj,
        (GraphSyncWork, MeshSyncWork, SkeletonSyncWork, PointSyncWork),
    )


class GraphSyncWork(sync.Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MeshSyncWork(sync.Mesh):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SkeletonSyncWork(sync.Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSyncWork(sync.Points):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@dataclasses.dataclass
class Link:
    """
    Represents the linkage mapping information.

    Parameters
    ----------
    mapping: Union[list[int], str]
        The mapping information between the source and target layers.
        The mapping will have the same length as the source vertices and the values will be the corresponding target vertex indices.
        If map_value_is_index is True, the mapping values should the dataframe index for target vertices (i.e. df.index), if False they are the positional indices (0-N).
        If a string, the mapping is the column name in the target's vertex dataframe.
    source: str
        The name of the source layer, typically the one with more vertices than the target. E.g. a graph or mesh to a skeleton, or a graph or skeleton to point annotations.
        If not provided, will be set to the layer name of the object this is being added to. Either source or target can be undefined, but not both.
    target: str
        The name of the target layer, typically the one with fewer vertices. E.g. a skeleton from a graph or mesh, or point annotations from a skeleton.
        If not provided, will be set to the layer name of the object this is being added to. Either source or target can be undefined, but not both.
    map_value_is_index: bool, optional
        If True, assumes the values in the list or the mapping are a non-positional dataframe index.
    """

    mapping: Union[list[int], str]
    source: Optional[str] = None
    target: Optional[str] = None
    map_value_is_index: bool = True

    def __post_init__(self):
        if isinstance(self.mapping, (list, np.ndarray)):
            self.mapping = np.array(self.mapping, dtype=int)
        if self.source is None and self.target is None:
            raise ValueError("Either source or target must be defined.")

    def mapping_to_index(self, vertex_data: pd.DataFrame) -> np.ndarray:
        "Map positional values to index values"
        if self.map_value_is_index:
            return self.mapping
        else:
            return vertex_data.index.values[self.mapping]
