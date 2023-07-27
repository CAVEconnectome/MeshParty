try:
    from .. import trimesh_vtk

    _vtk_loaded = True
except:
    _vtk_loaded = False

from ..skeleton_io import swc_node_labels
from ..trimesh_io import Mesh
from ..skeleton import Skeleton, resample

import contextlib
import io
import pandas as pd
import numpy as np
from scipy import sparse
from functools import wraps
from .utils import (
    DEFAULT_VOXEL_RESOLUTION,
    MeshworkIndexFactory,
    in1d_items,
    in1d_first_item,
    unique_column_name,
    compress_mesh_data,
    decompress_mesh_data,
    MaskedMeshMemory,
    window_matrix,
)
from . import meshwork_io


class AnchoredAnnotationManager(object):
    def __init__(
        self,
        anchor_mesh=None,
        filter_mesh=None,
        voxel_resolution=None,
    ):
        """Collection of dataframes anchored to a common mesh and filter.

        Parameters
        ----------
        mesh : trimesh_io.Mesh, optional
            Mesh to use to link points to vertex indices, by default None
        voxel_resolution : array-like, optional
            3-element array of the resolution of voxels in the point columns of the dataframes, by default None
        """
        if voxel_resolution is None:
            voxel_resolution = DEFAULT_VOXEL_RESOLUTION
        self._voxel_resolution = np.array(voxel_resolution).reshape((1, 3))
        if anchor_mesh is None:
            self._anchor_mesh = None
        else:
            self._anchor_mesh = MaskedMeshMemory(anchor_mesh)

        if filter_mesh is None:
            filter_mesh = self._anchor_mesh
        self._filter_mesh = MaskedMeshMemory(filter_mesh)

        self._MeshIndex = np.array
        self._data_tables = dict()

    def __len__(self):
        return len(self._data_tables)

    def __getitem__(self, key):
        return self._data_tables[key]

    def __delitem__(self, key):
        if key in self._data_tables:
            del self._data_tables[key]
            del self.__dict__[key]

    def __repr__(self):
        return f"Data tables: {self.table_names}"

    def items(self):
        return self._data_tables.items()

    @property
    def table_names(self):
        """List of data table names"""
        return list(self._data_tables.keys())

    @property
    def voxel_resolution(self):
        """Resolution in nm of the data in the point column"""
        return self._voxel_resolution

    @voxel_resolution.setter
    def voxel_resolution(self, new_res):
        self._voxel_resolution = np.array(new_res).reshape((1, 3))
        for tn in self.table_names:
            self._data_tables[tn].voxel_resolution = self._voxel_resolution

    def update_anchor_mesh(self, new_mesh):
        """Change or add the mesh to use for proximity-linking"""

        self._anchor_mesh = new_mesh

        for name in self.table_names:
            table = self._data_tables[name]
            if table.anchored is False:
                continue

            if table._index_column_base in table._original_columns:
                index_column = table._index_column_base
            else:
                index_column = None

            if table._point_column in table._original_columns:
                point_column = table._point_column
            else:
                point_column = None

            self._data_tables[name] = AnchoredAnnotation(
                name,
                table.data_original,
                self._anchor_mesh,
                point_column=point_column,
                max_distance=table._max_distance,
                index_column=index_column,
                voxel_resolution=self.voxel_resolution,
            )

            self._data_tables[name]._filter_data(self._filter_mesh)

    def _add_attribute(self, key):
        if key in dir(self):
            if not isinstance(self.__getattribute__(key), AnchoredAnnotation):
                return
        else:
            self.__dict__[key] = self._data_tables[key]

    def add_annotations(
        self,
        name,
        data,
        mask=False,
        point_array=False,
        anchored=True,
        point_column=None,
        max_distance=np.inf,
        index_column=None,
        overwrite=False,
        voxel_resolution=None,
    ):
        "Add a dataframe to the manager"

        if name in self.table_names and overwrite is False:
            raise ValueError(
                "Table name already taken. Overwrite or choose a different name."
            )
        else:
            self.remove_annotations(name)

        if voxel_resolution is None:
            voxel_resolution = self.voxel_resolution

        if issubclass(type(data), np.ndarray):
            if len(data.shape) == 2:
                if data.shape[1] == 3:
                    data = pd.DataFrame(data={"pt": data.tolist()})
                    point_column = "pt"
                else:
                    raise ValueError("Arrays must be of size Nx3")

        self._data_tables[name] = AnchoredAnnotation(
            name,
            data,
            self._filter_mesh,
            point_column=point_column,
            mask=mask,
            point_array=point_array,
            anchor_to_mesh=anchored,
            max_distance=max_distance,
            index_column=index_column,
            voxel_resolution=voxel_resolution,
        )
        self._data_tables[name]._register_MeshIndex(self._MeshIndex)
        self._data_tables[name]._filter_data(self._filter_mesh)

        self._add_attribute(name)

    def remove_annotations(self, name):
        "Remove a data table from the manager"
        if isinstance(name, str):
            name = [name]
        for n in name:
            del self[n]

    def anchor_annotations(self, name):
        "If an annotation is not anchored, link it to the current anchor mesh and apply the current filters"
        if isinstance(name, str):
            name = [name]
        for n in name:
            self._data_tables[n]._anchor_to_mesh(self._anchor_mesh)
            self._data_tables[n]._filter_data(self._filter_mesh)

    def filter_annotations(self, new_mesh):
        "Use a masked mesh to filter all anchored annotations"
        self._filter_mesh = MaskedMeshMemory(new_mesh)
        for tn in self.table_names:
            self._data_tables[tn]._filter_data(self._filter_mesh)

    def remove_filter(self):
        "Remove filters from the annotations"
        self._filter_mesh = self._anchor_mesh
        for tn in self.table_names:
            self._data_tables[tn]._reset_filter()

    def _register_MeshIndex(self, NewMeshIndex):
        self._MeshIndex = NewMeshIndex
        for tn in self.table_names:
            self._data_tables[tn]._register_MeshIndex(NewMeshIndex)


class AnchoredAnnotation(object):
    def __init__(
        self,
        name,
        data,
        mesh=None,
        mask=False,
        point_array=False,
        anchor_to_mesh=True,
        point_column=None,
        max_distance=np.inf,
        index_column=None,
        voxel_resolution=None,
        voxel_scaling=None,
    ):
        if point_array is True:
            if isinstance(data, np.ndarray):
                data = data.tolist()
            if point_column is None:
                point_column = "position"
            data = pd.DataFrame({point_column: data})
        elif mask == True:
            if isinstance(data, np.ndarray):
                data = data.ravel()
            index_column = "mesh_index"
            data = pd.DataFrame({index_column: data})

        self._name = name
        self._data = data.reset_index(drop=True)
        self._original_columns = data.columns
        self._max_distance = max_distance

        self._point_column = point_column
        if index_column is None:
            defined_index = False
            index_column = unique_column_name(None, "mesh_index", data)
        else:
            defined_index = True
        self._defined_index = defined_index
        self._index_column_base = index_column
        self._index_column_filt = f"{index_column}_filt"

        # Initalize to -1 so the column exists
        self._data[self._index_column_base] = -1
        self._data[self._index_column_filt] = -1

        if self._defined_index:
            if mesh is not None and anchor_to_mesh is True:
                self._data[self._index_column_base] = mesh.map_indices_to_unmasked(
                    data[index_column]
                )
                self._data[self._index_column_filt] = data[index_column].values

        self._orig_col_plus_index = list(self._original_columns) + [
            self._index_column_filt
        ]

        valid_column = unique_column_name(index_column, "valid", data)
        self._data[valid_column] = True
        self._valid_column = valid_column

        self._mask_column = unique_column_name(index_column, "in_mask", data)
        # Initalize in_mask to True before any subsequent masking
        self._data[self._mask_column] = True

        if voxel_resolution is None:
            voxel_resolution = DEFAULT_VOXEL_RESOLUTION
        if mesh.voxel_scaling is not None:
            voxel_resolution = voxel_resolution * mesh.voxel_scaling

        self._voxel_resolution = np.array(voxel_resolution).reshape((1, 3))
        self._MeshIndex = None

        self._anchor_mesh = None
        self._filter_mesh = None
        self._anchored = anchor_to_mesh
        if self._anchored and mesh is not None:
            self._anchor_points(mesh)

    def __repr__(self):
        return self.df.__repr__()

    def __getitem__(self, key):
        return self.df.__getitem__(key)

    def __len__(self):
        return len(self.df)

    @property
    def name(self):
        return self._name

    @property
    def point_column(self):
        return self._point_column

    @property
    def index_column(self):
        return self._index_column_filt

    @property
    def index_column_original(self):
        return self._index_column_base

    @property
    def _is_valid(self):
        return self._data[self._valid_column]

    @property
    def _in_mask(self):
        return self._data[self._mask_column]

    @property
    def _is_included(self):
        return np.logical_and(self._is_valid, self._in_mask)

    @property
    def df(self):
        if self.anchored:
            return self._data[self._orig_col_plus_index][self._is_included]
        else:
            return self._data[self._original_columns][self._is_included]

    @property
    def voxel_resolution(self):
        return self._voxel_resolution

    @voxel_resolution.setter
    def voxel_resolution(self, new_res):
        self._voxel_resolution = np.array(new_res).reshape((1, 3))

    @property
    def voxels(self):
        if self.point_column is None or len(self.df) == 0:
            return np.zeros((0, 3))
        else:
            return np.vstack(self.df[self.point_column].values)

    @property
    def points(self):
        if self.point_column is None:
            return np.zeros((0, 3))
        else:
            return self.voxels * self.voxel_resolution

    def _register_MeshIndex(self, NewClass):
        self._MeshIndex = NewClass

    @property
    def MeshIndex(self):
        if self._MeshIndex is None:
            return np.array
        return self._MeshIndex

    @property
    def mesh_index(self):
        if self.anchored:
            return self.MeshIndex(
                self._data[self._index_column_filt][self._is_included].values
            )
        else:
            return None

    @property
    def skel_index(self):
        return self.mesh_index.to_skel_index

    @property
    def mesh_mask(self):
        return self.mesh_index.to_mesh_mask

    @property
    def skel_mask(self):
        return self.mesh_index.to_skel_mask

    @property
    def _mesh_index_base(self):
        return self._data[self._index_column_base].values

    @property
    def data_original(self):
        return self._data[self._original_columns]

    def _anchor_points(self, mesh):
        if self._defined_index is False:
            dist, minds_filt = mesh.kdtree.query(self.points)
            self._data[self._index_column_filt] = minds_filt

            minds_base = mesh.map_indices_to_unmasked(minds_filt)
            self._data[self._index_column_base] = minds_base
            self._data[self._valid_column] = dist < self._max_distance
        else:
            self._data[self._valid_column] = True

        self._anchor_mesh = MaskedMeshMemory(mesh, index_only=True)
        self._filter_mesh = self._anchor_mesh

    def _filter_data(self, filter_mesh):
        """Get the subset of data points that are associated with the mesh"""
        if self._anchored:
            self._data[self._mask_column] = filter_mesh.node_mask[self._mesh_index_base]
            self._data[
                self._index_column_filt
            ] = filter_mesh.filter_unmasked_indices_padded(self._mesh_index_base)
            self._filter_mesh = MaskedMeshMemory(filter_mesh)

    def _filter_query_response(self, row_filter):
        _parent = self

        class _FilterQueryResponse(object):
            def __init__(self, row_filter):
                self._row_filter = row_filter

            @property
            def row_filter(self):
                return self._row_filter

            @property
            def voxels(self):
                return _parent.voxels[self.row_filter]

            @property
            def points(self):
                return _parent.points[self.row_filter]

            @property
            def df(self):
                return _parent.df[self.row_filter]

            @property
            def count(self):
                return np.sum(row_filter)

            @property
            def mesh_index(self):
                return _parent.MeshIndex(
                    _parent._data[_parent._index_column_filt][self.row_filter].values
                )

        return _FilterQueryResponse(row_filter)

    def _filter_query(self, node_mask):
        """Returns the data contained with a given filter without changing any indexing."""
        node_mask_base = self._filter_mesh.map_boolean_to_unmasked(node_mask)
        if self._anchored:
            keep_rows = node_mask_base[self._mesh_index_base]
            return keep_rows[self._is_included]
        else:
            return np.full(len(self.df), True)

    def filter_query(self, node_mask):
        row_filter = self._filter_query(node_mask)
        return self._filter_query_response(row_filter)

    def _reset_filter(self):
        if self._anchored:
            self._data[self._mask_column] = True
            self._data[
                self._index_column_filt
            ] = self._anchor_mesh.filter_unmasked_indices(self._mesh_index_base)
            self._filter_mesh = self._anchor_mesh

    def _anchor_to_mesh(self, anchor_mesh):
        self._anchored = True
        self._reset_filter()
        self._data[self._valid_column] = True
        self._anchor_points(anchor_mesh)

    @property
    def anchored(self):
        return self._anchored


class Meshwork(object):
    """Class to integrate mesh, skeleton, and annotation data

    Parameters
    ----------
    mesh : trimesh_io.Mesh
        Mesh in a form for skeletonization and attaching synapses
    skeleton : skeleton.Skeleton, optional
        Skeleton built from the same mesh as Mesh.
    seg_id : int or None, optional
        Segment id for object
    voxel_resolution : array-like or None, optional
        3-element voxel resolution (if None, [4,4,40]) to convert between
        annotation point units (voxels) and mesh vertex units (e.g. nanometers).
    """

    def __init__(
        self,
        mesh=None,
        skeleton=None,
        seg_id=None,
        voxel_resolution=None,
        voxel_scaling=None,
    ):
        if mesh is None and skeleton is None:
            raise ValueError("Must include at least one of mesh or skeleton")

        if mesh is None and skeleton is not None:
            mesh = Mesh(
                vertices=skeleton.vertices,
                faces=np.zeros((0, 3)),
                link_edges=skeleton.edges,
            )

        self._seg_id = seg_id
        self._mesh = mesh
        self._skeleton = skeleton

        if voxel_resolution is None:
            voxel_resolution = DEFAULT_VOXEL_RESOLUTION
        self._anno = AnchoredAnnotationManager(
            self._mesh, voxel_resolution=voxel_resolution
        )

        self._voxel_scaling = voxel_scaling
        self._mesh.voxel_scaling = voxel_scaling
        if skeleton is not None:
            self._skeleton.voxel_scaling = voxel_scaling

        self._original_mesh_data = None
        self._MeshIndex = None
        self._SkeletonIndex = None
        self._recompute_indices()

    @property
    def seg_id(self):
        """Segmentation id for the object"""
        return self._seg_id

    @property
    def voxel_scaling(self):
        return self._voxel_scaling

    @voxel_scaling.setter
    def voxel_scaling(self, new_scaling):
        self._mesh.voxel_scaling = new_scaling
        if self.skeleton is not None:
            self._skeleton.voxel_scaling = new_scaling
        self._recompute_indices()

    def __copy__(self):
        with io.BytesIO() as bio:
            self.save_meshwork(bio)
            bio.seek(0)
            nrn_copy = load_meshwork(bio)
        return nrn_copy

    def copy(self):
        return self.__copy__()

    ##################
    # Mesh functions #
    ##################

    def _recompute_indices(self):
        self._MeshIndex, self._SkeletonIndex = MeshworkIndexFactory(self)
        self.anno._register_MeshIndex(self._MeshIndex)
        if self.skeleton is not None:
            self.skeleton._register_skeleton_index(self._SkeletonIndex)

    @property
    def MeshIndex(self):
        """Numpy array-like object for mesh indices with handy conversion features"""
        if self._MeshIndex is None:
            self._recompute_indices()
        return self._MeshIndex

    def _convert_to_meshindex(self, mesh_indices):
        if isinstance(mesh_indices, self.MeshIndex):
            return mesh_indices
        elif isinstance(mesh_indices, self.SkeletonIndex):
            return mesh_indices.to_mesh_index
        else:
            if np.isscalar(mesh_indices):
                mesh_indices = np.array([mesh_indices], dtype=int)
            return self.MeshIndex(mesh_indices)

    @property
    def SkeletonIndex(self):
        """Numpy array-like object for skeleton indices with handy conversion features"""
        if self._SkeletonIndex is None:
            self._recompute_indices()
        return self._SkeletonIndex

    def _convert_to_skelindex(self, skel_indices):
        if isinstance(skel_indices, self.SkeletonIndex):
            return skel_indices
        elif isinstance(skel_indices, self.MeshIndex):
            return skel_indices.to_skel_index
        else:
            if np.isscalar(skel_indices):
                skel_indices = np.array([skel_indices], dtype=int)
            return self.SkeletonIndex(skel_indices)

    def _reset_indices(self):
        self._MeshIndex = None
        self._SkeletonIndex = None
        if self.skeleton is not None:
            self.skeleton._register_skeleton_index(self.SkeletonIndex)
        self.anno._register_MeshIndex(self.MeshIndex)

    @property
    def mesh(self):
        """Copy of the neuronal mesh"""
        return self._mesh

    @property
    def mesh_mask(self):
        """The mesh mask, a boolean array the length of the number of mesh vertices."""
        return self.mesh.node_mask

    @property
    def vertices(self):
        return self.mesh.vertices

    @property
    def edges(self):
        return self.mesh.edges

    @property
    def faces(self):
        return self.mesh.faces

    def apply_mask(self, mask):
        """Apply a mesh mask to the meshwork object

        Parameters
        ----------
        mask : array of booleans
            Array with the same number of elements as mesh vertices. True elements are kept,
            False elements are masked out.
        """
        if self._original_mesh_data is None:
            self._original_mesh_data = compress_mesh_data(self.mesh)

        if self.skeleton is not None:
            sk_mask = self._mesh_mask_to_skel_mask(mask)
            self.skeleton.apply_mask(sk_mask, in_place=True)

        self._mesh = self.mesh.apply_mask(mask)
        self._anno.filter_annotations(self.mesh)
        self._reset_indices()

    def reset_mask(self, to=None):
        """Remove mask and restore object to its original state.

        Parameters
        ----------
        to : array, optional
            Mask array of booleans to apply after resetting the mask. Equivalent to reset_mask followed by apply_mask.
        """
        if self._original_mesh_data is not None:
            self._anno.remove_filter()

            vs, fs, es, nm, vxsc = decompress_mesh_data(*self._original_mesh_data)
            self._mesh = Mesh(vs, fs, link_edges=es, node_mask=nm, voxel_scaling=vxsc)

            self._original_mesh_data = None
            if self.skeleton is not None:
                self._skeleton.reset_mask(in_place=True)
            self._reset_indices()

            if to is not None:
                self.apply_mask(to)

    @contextlib.contextmanager
    def mask_context(self, mask):
        """Use with-statement context to temporarily act on a masked state of an object.

        Parameters
        ----------
        mask : array or None,
            A boolean array with the same number of elements as mesh vertices. True elements are kept, False are masked out. If None, resets the mask entirely.
        """
        current_mask = self.mesh_mask
        try:
            if mask is None:
                self.reset_mask()
            else:
                self.apply_mask(mask)
            yield self
        finally:
            self.reset_mask(to=current_mask)

    @property
    def mesh_indices(self):
        return self.MeshIndex(np.arange(self.mesh.n_vertices))

    ##################
    # Anno functions #
    ##################

    @property
    def anno(self):
        """AnchoredAnnotationManager associated with the anchor mesh"""
        return self._anno

    def add_annotations(
        self,
        name,
        data,
        anchored=True,
        mask=False,
        point_array=False,
        point_column=None,
        max_distance=np.inf,
        index_column=None,
        overwrite=False,
    ):
        """Associate annotation data with the mesh

        Parameters
        ----------
        name : str
            Name of the annotation data, to be used as a unique key.
        data : pandas.DataFrame
            DataFrame containing annotation data.
        anchored : bool, optional
            If True, associates locations in a specified point column with mesh vertices.
        mask : bool, optional
            If True, assumes the data is a boolean with the same length as the mesh vertices and converts True values to mesh indices. The resulting annotation does not keep track of points, only indices.
        point_array : bool, optional
            If True, assumes the data is a list or array of 3d points and constructs a basic dataframe to hold them (using the point_column value if provided).
        point_column : str or None, optional
            Column name holding 3-element point position in voxel units. Must be specified for anchored annotations.
        max_distance : numeric, optional
            Maximum distance between annotation point and mesh vertex to connect. If a row is
            farther than this distance, it is silently filtered out of the annotation data. By default, np.inf.
        index_column : str or None, optional
            If mesh indices are already computed, entries here are used instead of closest-distance.
        overwrite : bool, optional
            If True, overwrite an existing annotation with the same name.
        """
        self._anno.add_annotations(
            name=name,
            data=data,
            anchored=anchored,
            mask=mask,
            point_array=point_array,
            point_column=point_column,
            max_distance=max_distance,
            index_column=index_column,
            overwrite=overwrite,
        )

    def remove_annotations(self, name):
        """Remove annotation from AnnotationManager

        Parameters
        ----------
        name : str
            Name of the annotation to remove
        """
        self._anno.remove_annotations(name)

    def anchor_annotations(self, name):
        """Set an unanchored annotation to an anchored one

        Parameters
        ----------
        name : str
            Name of the annotation data to anchor
        """
        self._anno.anchor_annotations(name)

    ######################
    # Skeleton functions #
    ######################

    class OnlyIfSkeleton(object):
        """Convenience class to handle situations without skeletons"""

        @staticmethod
        def exists(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                if self.skeleton is not None:
                    return func(self, *args, **kwargs)
                else:
                    return None

            return wrapper

    @property
    def skeleton(self):
        """Associated skeleton"""
        return self._skeleton

    @property
    @OnlyIfSkeleton.exists
    def skeleton_indices(self):
        return self.SkeletonIndex(np.arange(self.skeleton.n_vertices))

    def skeletonize_mesh(
        self,
        soma_pt=None,
        collapse_soma=True,
        soma_thresh_distance=7500,
        invalidation_distance=12000,
        compute_radius=True,
        shape_function="single",
        overwrite=False,
        collapse_function="sphere",
        collapse_params={},
    ):
        """Skeletonize the anchor mesh. Always uses the mesh as used to initialize the class instance.

        Parameters
        ----------
        soma_pt : arrary or None, optional
            3-element point (in mesh coordinates) specifying the center of the soma.
        soma_thresh_distance : numeric, optional
            Distance threshold to consider all inside the soma. Default is 7500.
        invalidation_distance : int, optional
            Invalidation distance for the TEASAR skeletonization. Default is 12000.
        compute_radius : bool, optional
            If True, computes the "shape distance function" giving the diameter of the mesh at the skeleton point.
        shape_function: 'single' or 'cone', optional
            Determines whether to use a single ray or a cone of rays to determine mesh diameter.
        overwrite : bool, optional
            If True, will run even if an existing skeletonization is present.
        """
        from meshparty.skeletonize import skeletonize_mesh

        if self._original_mesh_data is not None:
            vs, fs, es, nm, vxsc = decompress_mesh_data(*self._original_mesh_data)
            mesh_to_sk = Mesh(vs, fs, link_edges=es, node_mask=nm, voxel_scaling=vxsc)
        else:
            mesh_to_sk = self.mesh

        if self._skeleton is None or overwrite is True:
            self._skeleton = skeletonize_mesh(
                mesh_to_sk,
                soma_pt=soma_pt,
                soma_radius=soma_thresh_distance,
                collapse_soma=collapse_soma,
                invalidation_d=invalidation_distance,
                compute_original_index=True,
                compute_radius=compute_radius,
                shape_function=shape_function,
                collapse_function=collapse_function,
                collapse_params=collapse_params,
                meta={"root_id": self.seg_id},
            )
            self._reset_indices()
        else:
            print("Skeleton already exists")
        pass

    # all functions of this group take filtered indices and return filtered indices.
    def _mind_to_skind_padded(self, minds):
        minds_b = self.mesh.map_indices_to_unmasked(minds)
        skinds = self.skeleton.mesh_to_skel_map[minds_b]
        return skinds

    def _mind_to_skind(self, minds):
        mind_padded = self._mind_to_skind_padded(minds)
        return mind_padded[mind_padded >= 0]

    def _mesh_mask_to_skel_mask(self, mesh_mask):
        mesh_mask = self.mesh.map_boolean_to_unmasked(mesh_mask)
        skel_inds = np.unique(self.skeleton.mesh_to_skel_map_base[mesh_mask])
        skel_mask = np.full(self.skeleton.unmasked_size, False)
        skel_mask[skel_inds] = True
        return skel_mask

    def _skind_to_mind_mask_base(self, skinds):
        skinds_b = self.skeleton.map_indices_to_unmasked(skinds)
        minds_b_assoc = np.isin(self.skeleton.mesh_to_skel_map_base, skinds_b)
        return minds_b_assoc

    def _skind_to_mind_mask(self, skinds):
        return self.mesh.filter_unmasked_boolean(self._skind_to_mind_mask_base(skinds))

    def _skind_to_mind_index(self, skinds):
        return np.flatnonzero(self._skind_to_mind_mask(skinds))

    def _skind_regions(self, skinds):
        out = in1d_items(self.skeleton.mesh_to_skel_map[self.mesh.node_mask], skinds)
        return out

    def _skind_region_first(self, skinds):
        if isinstance(skinds, int):
            skinds = [skinds]
        if issubclass(type(skinds), np.ndarray):
            if len(skinds.shape) == 0:
                skinds = skinds.reshape(1)
        return in1d_first_item(
            self.skeleton.mesh_to_skel_map[self.mesh.node_mask], skinds
        )

    @OnlyIfSkeleton.exists
    def skeleton_property_to_mesh(
        self, skeleton_property, mesh_inds=None, no_map_value=-1.0
    ):
        """Map a property at skeleton points to the mesh, converting nodes
        without a clear mapping to a specified value (default -1).
        """
        if mesh_inds is None:
            mesh_inds = np.arange(self.mesh.n_vertices)
        mesh_inds = self._convert_to_meshindex(mesh_inds)
        mesh_property = np.full(len(mesh_inds), no_map_value)
        skids = mesh_inds.to_skel_index_padded
        mesh_property[skids >= 0] = skeleton_property[skids[skids >= 0]]
        return mesh_property

    @OnlyIfSkeleton.exists
    def mesh_property_to_skeleton(
        self,
        mesh_property,
        aggfunc="mean",
    ):
        """Map a property at all mesh vertices to skeletons, aggregating mesh vertices as desired.

        Parameters
        ----------
        mesh_property : array
            N-element array with same length as mesh vertices
        aggfunc : str or function, optional
            Either a pre-defined aggregation function or one of "mean", "median", "max, or "min". By default 'mean'.

        Returns
        -------
        np.ndarray
            Array of aggregated values
        """
        if aggfunc == "mean":
            aggfunc = np.mean
        elif aggfunc == "median":
            aggfunc = np.median
        elif aggfunc == "max":
            aggfunc = np.max
        elif aggfunc == "min":
            aggfunc = np.min
        elif isinstance(aggfunc, str):
            raise ValueError(
                "Only string values allowed are 'mean', 'median', 'max', and 'min'."
            )

        mesh_property = np.array(mesh_property)
        arr_grouped = self.skeleton_indices.to_mesh_region
        return np.array([aggfunc(mesh_property[x]) for x in arr_grouped])

    @property
    @OnlyIfSkeleton.exists
    def branch_points_skel(self):
        """Skeleton index of branch points"""
        return self.SkeletonIndex(self.skeleton.branch_points)

    @property
    @OnlyIfSkeleton.exists
    def branch_points_region(self):
        """List of arrays of all mesh indices associated with each branch point."""
        return self.branch_points_skel.to_mesh_region

    @property
    @OnlyIfSkeleton.exists
    def branch_points(self):
        """Array with one mesh point per skeleton branch point."""
        return self.branch_points_skel.to_mesh_region_point

    @property
    @OnlyIfSkeleton.exists
    def end_points_skel(self):
        """Skeleton index of each end point"""
        return self.SkeletonIndex(self.skeleton.end_points)

    @property
    @OnlyIfSkeleton.exists
    def end_points(self):
        """Array with one mesh point per skeleton end point."""
        return self.end_points_skel.to_mesh_region_point

    @property
    @OnlyIfSkeleton.exists
    def end_points_region(self):
        """List of arrays of all mesh indices associated with each end point."""
        return self.end_points_skel.to_mesh_region

    @property
    @OnlyIfSkeleton.exists
    def root_skel(self):
        """Skeleton index of the root node (as a single element array)"""
        return self.SkeletonIndex([self.skeleton.root])

    @property
    @OnlyIfSkeleton.exists
    def root_region(self):
        """Array of all mesh indices associated with the root node."""
        return self.root_skel.to_mesh_region[0]

    @property
    @OnlyIfSkeleton.exists
    def root(self):
        """Mesh index of one (arbitrary) point associated with the root node."""
        return self.root_skel.to_mesh_region_point[0]

    @OnlyIfSkeleton.exists
    def parent_index(self, mesh_inds, include_parent_free=False, return_as_skel=False):
        """Given a list of mesh indicies, get mesh indices of parents.

        Parameters
        ----------
        mesh_inds : array or int
            Mesh indices to check
        include_parent_free : bool, optional
            If True, returns one element for each mesh index. Nodes with no parents have a -1.
            If False (default), returns a list of elements that omits elements without parents,
            breaking a one-to-one correspondance with the inputs.
        return_as_skel : bool, optional
            Return as skeleton indices. This effectively includes parent-free nodes as well.

        Returns
        -------
        array
            Collection of parent indices in the desired form
        """
        mesh_inds = self._convert_to_meshindex(mesh_inds)
        parent_index = self.skeleton.parent_nodes(mesh_inds.to_skel_index)
        if return_as_skel:
            return parent_index
        if include_parent_free:
            return parent_index.to_mesh_region_point
        else:
            return parent_index[parent_index >= 0].to_mesh_region_point

    @OnlyIfSkeleton.exists
    def child_index(self, mesh_inds, return_as_skel=False):
        """Given a list of mesh indicies, get list of mesh indices of child nodes.

        Parameters
        ----------
        mesh_inds : array or int
            Mesh indices to check
        return_as_skel : bool, optional
            Return as skeleton indices.

        Returns
        -------
        list
            List of arrays of child indices in the desired form
        """
        if np.isscalar(mesh_inds):
            return_scalar = True
        else:
            return_scalar = False
        mesh_inds = self._convert_to_meshindex(mesh_inds)
        child_index = self.skeleton.child_nodes(mesh_inds.to_skel_index)
        if return_as_skel:
            return child_index
        if return_scalar:
            return child_index[0].to_mesh_region_point
        return self.MeshIndex([n.to_mesh_region_point for n in child_index])

    @OnlyIfSkeleton.exists
    def jump_proximal(self, ind, include_initial=False, hops=1):
        """Find the next branch point (or root) towards root from a given index.

        Parameters
        ----------
        ind : int or MeshIndex
            Mesh index t
        include_initial : bool, optional
            If False, ind is never included in the list of branch points toward root.
            Default is False.
        hops : int, optional
            Number of hops toward root to take. Default is 1.

        Returns
        -------
        mesh_index
            Index of the proximal branch point/root after the desired number of hops.
        """
        ind = self._convert_to_meshindex(ind)
        skind = ind.to_skel_index[0]

        topo_points = self.skeleton.topo_points
        if include_initial == False:
            if skind != self.skeleton.root:
                topo_points = topo_points[topo_points != ind]

        ptr = self.skeleton.path_to_root(skind)
        is_topo = np.isin(ptr, self.skeleton.topo_points)
        topo_pts_on_path = ptr[np.flatnonzero(is_topo)]
        hop_ind = np.min((len(topo_pts_on_path), hops)) - 1
        skind_out = self.SkeletonIndex(topo_pts_on_path[int(hop_ind)])
        return skind_out.to_mesh_region_point

    @OnlyIfSkeleton.exists
    def jump_distal(self, ind):
        """Finds the next topologically interesting (branch or end point) away from root starting from an initial mesh index.

        Parameters
        ----------
        ind : Mesh index
            Index of the initial mesh to use for

        Returns
        -------
        mesh_index
            Index of a mesh point corresponding to the jump.
        """
        ind = self._convert_to_meshindex(ind)
        skind = ind.to_skel_index
        if skind in np.concatenate(
            (self.skeleton.end_points, self.skeleton.branch_points)
        ):
            return ind  # If it's an end point, return immediately

        d = sparse.csgraph.dijkstra(
            self.skeleton.csgraph.T, directed=True, indices=skind
        )

        topo_pts = self.skeleton.topo_points
        proximal_pt = topo_pts[np.argmin(d[0, topo_pts])]
        return self.SkeletonIndex(proximal_pt).to_mesh_region_point

    @OnlyIfSkeleton.exists
    def distance_to_root(self, mesh_indices=None):
        """Distance to root for mesh indices along skeleton.

        Parameters
        ----------
        mesh_indices : array-like
            Collection of mesh indices to look up

        Returns
        -------
        array
            Array of distances to root measured along the skeleton. If no corresponding
            skeleton index exists for a mesh point, a NaN is used.
        """
        if mesh_indices is None:
            mesh_indices = np.arange(len(self.mesh.vertices))
        mesh_indices = self._convert_to_meshindex(mesh_indices)
        ds = np.full(len(mesh_indices), np.nan)
        skinds = mesh_indices.to_skel_index_padded
        ds[skinds >= 0] = self.skeleton.distance_to_root[skinds[skinds >= 0]]
        return ds

    @OnlyIfSkeleton.exists
    def downstream_of(self, mesh_index, return_as_skel=False):
        """Get mesh indices distal to a collection of mesh indices

        Parameters
        ----------
        mesh_index : int or array-like
            Mesh index or collection of mesh indices
        return_as_skel : bool, optional
            If True, returns downstream indices as SkeletonIndex. Default is False.

        Returns
        -------
        array or list
            If a single index was given, returns an array of mesh indices downstream.
            If a collection of indices were given, returns a list of arrays of mesh indices downstream.
        """
        if np.isscalar(mesh_index):
            use_scalar = True
            mesh_index = [mesh_index]
        else:
            use_scalar = False
        mesh_index = self._convert_to_meshindex(mesh_index)
        skinds_downstream = self.skeleton.downstream_nodes(mesh_index.to_skel_index)
        if return_as_skel:
            return skinds_downstream
        minds_downstream = []
        for ds_list in skinds_downstream:
            minds_downstream.append(ds_list.to_mesh_index)
        if use_scalar:
            minds_downstream = minds_downstream[0]
        return minds_downstream

    def _get_segment_list(self, seg_inds, return_as_skel):
        segment_list = []
        for seg in seg_inds:
            if return_as_skel is False:
                segment_list.append(self.skeleton.segments[seg].to_mesh_index_ordered)
            else:
                segment_list.append(self.skeleton.segments[seg])
        return segment_list

    @OnlyIfSkeleton.exists
    def same_segment(self, mesh_inds, return_as_skel=False):
        """Get all indices within the same segment (region between branch/end points)

        Parameters
        ----------
        mesh_inds : int or array-like
            Mesh index or collection of mesh indices
        return_as_skel : bool, optional
            If True, return as SkeletonIndex. Default is False.

        Returns
        -------
        array or list
            If a single index was given, returns an array of mesh indices in segment.
            If a collection of indices were given, returns a list of arrays of mesh indices in segment.
        """
        if np.isscalar(mesh_inds):
            return_scalar = True
        else:
            return_scalar = False

        mesh_inds = self._convert_to_meshindex(mesh_inds)
        segs = self.skeleton.segment_map[mesh_inds.to_skel_index]
        segment_list = self._get_segment_list(segs, return_as_skel)

        if return_scalar:
            segment_list = segment_list[0]
        return segment_list

    @OnlyIfSkeleton.exists
    def segments(self, return_as_skel=False):
        """Get all segments (regions between topological points)

        Parameters
        ----------
        return_as_skel : bool, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        return self._get_segment_list(
            np.arange(len(self.skeleton.segments)), return_as_skel
        )

    @OnlyIfSkeleton.exists
    def same_branch(self, ind, hops=1, extend_to_root=False, return_as_skel=False):
        """Get all indices on the same segment (or, optionally, primary branch)
        and all downstream nodes.

        Parameters
        ----------
        inds : int or SkeletonIndex or MeshIndex
            Mesh index to base branch index off of. Must not be an index of the root.
        hops : int, optional
            Number of branch points to hop towards root before finding downstream vertices, by default 1.
        extend_to_root : bool, optional
            If True, starts the branch at the root intead of a branch point, by default False
        return_as_skel : bool, optional
            If True, returns skeleton indices instead of mesh indices, by default False

        Returns
        -------
        branch_inds : MeshIndex or SkeletonIndex
            Indices on the same branch (SkeletonIndex if return_as_skel==True)
        """
        ind = self._convert_to_meshindex(ind)

        if ind in self.root_region:
            raise ValueError("Index is in root, branch is not defined")

        skind = ind.to_skel_index
        if extend_to_root:
            base_ind = self.root
        else:
            base_ind = self.jump_proximal(ind, hops=hops)

        ptb = self.path_between(ind, base_ind, return_as_skel=True)
        source_ind = self.SkeletonIndex(ptb[-2])
        return self.downstream_of(
            source_ind.to_mesh_region_point[0], return_as_skel=return_as_skel
        )

    def _distance_between(self, inds_source, inds_target, graph, squeeze):
        ds = sparse.csgraph.dijkstra(graph, directed=False, indices=inds_source)
        if squeeze:
            return ds[:, inds_target].squeeze()
        else:
            return ds[:, inds_target]

    @OnlyIfSkeleton.exists
    def distance_between(self, inds_source, inds_target, along_path=True, squeeze=True):
        """Get distance matrix between source and target mesh indices along the object

        Parameters
        ----------
        inds_source : int or array
            mesh indices for one side of the paths
        inds_target : int or array
            mesh indices for the other side of the paths
        along_path : bool, optional
            If True (default), use the skeleton
        squeeze : bool, optional
            If True (default), squeezes singlet dimensions of the distance matrix.
        Returns
        -------
        array
            Array of distances with shape (len(inds_source), len(inds_target)) before optional squeezing.
        """
        inds_source = self._convert_to_meshindex(inds_source)
        inds_target = self._convert_to_meshindex(inds_target)

        if along_path:
            return self._distance_between(
                inds_source.to_skel_index_padded,
                inds_target.to_skel_index_padded,
                self.skeleton.csgraph,
                squeeze=False,
            )
        else:
            return self._distance_between(
                inds_source, inds_target, self.mesh.csgraph, squeeze=squeeze
            )

    @OnlyIfSkeleton.exists
    def path_between(self, source_index, target_index, return_as_skel=False):
        """Get indices of a path between a source and a target mesh index.

        Parameters
        ----------
        source_index : int
            Mesh index of a source
        target_index : int
            Mesh index of a target
        return_as_skel : bool, optional
            If True, return as skeleton indices. Default is False.

        Returns
        -------
        array
            All indices between source and target.
        """
        source_index = self._convert_to_meshindex(source_index)
        target_index = self._convert_to_meshindex(target_index)
        skpath = self.SkeletonIndex(
            self.skeleton.path_between(
                source_index.to_skel_index[0], target_index.to_skel_index[0]
            )
        )
        if return_as_skel:
            return skpath
        return skpath.to_mesh_index_ordered

    def _within_distance(self, inds, graph, max_distance):
        ds = sparse.csgraph.dijkstra(graph, indices=inds, directed=False)
        return ds < max_distance

    @OnlyIfSkeleton.exists
    def within_distance(
        self, source_inds, distance, collapse=True, return_as_skel=False
    ):
        """Mesh indices within a distance along the object's skeleton

        Parameters
        ----------
        source_inds : array or int
            Source indices of the mesh
        distance : numeric
            Distance to query
        collapse : bool, optional
            If True, returns a mask that is True if index is close to any source. Default is True.
        return_as_skel : bool, optional
            If True, returns skeleton indices. Default is False.

        Returns
        -------
        array or list of arrays
            Mask (if return_as_skel is False) or skeleton index array.
            If collapse is True, a single array that is True if True for any source.
            If collapse is False, a list of arrays for each source.
        """
        if np.isscalar(source_inds) or collapse:
            return_scalar = True
        else:
            return_scalar = False

        source_inds = self._convert_to_meshindex(source_inds)

        dmask = self._within_distance(
            source_inds.to_skel_index, self.skeleton.csgraph, distance
        )
        if collapse:
            if (dmask.shape) == 1:
                dmask = dmask.reshape(1, -1)
            dmask = np.any(dmask, axis=0).reshape(1, -1)
            return_scalar = True
        if return_as_skel:
            if return_scalar:
                dmask = self.SkeletonIndex(np.flatnonzero(dmask))
            else:
                dmask = [self.SkeletonIndex(np.flatnonzero(m)) for m in dmask]
        else:
            dmask = dmask[:, self.skeleton.mesh_to_skel_map]
            if return_scalar:
                dmask = self.MeshIndex(np.flatnonzero(dmask))
            else:
                dmask = [self.MeshIndex(np.flatnonzero(m)) for m in dmask]
        return dmask

    @OnlyIfSkeleton.exists
    def path_length(self, inds=None):
        """Get path length of collection of mesh indices

        Parameters
        ----------
        inds : array-like
            Mesh indices to compute path length for. Can be in any order.

        Returns
        -------
        path_length : float
            Path length in mesh units.
        """
        if inds is None:
            return self.skeleton.path_length()
        inds = self._convert_to_meshindex(inds)
        return self.skeleton.path_length(inds.to_skel_index)

    @OnlyIfSkeleton.exists
    def total_path_length(self):
        return self.skeleton.csgraph.sum()

    @OnlyIfSkeleton.exists
    def linear_density(
        self,
        inds,
        width,
        weight=None,
        normalize=True,
        exclude_root=False,
    ):
        """Compute a sliding window average linear density of points across the object

        Parameters
        ----------
        inds : array
            Mesh indices for density (e.g. synapse locations).
        width : numeric
            width of average window (in all directions).
        weight : array None, optional
            Weight for each point for weighted average. If None, assumes weight of unity.
        normalize : bool, optional
            If False, sums the weights but does not normalize by amount of cable.
            Default is True.
        exclude_root : bool, optional
            If True, disconnects root from the graph for the case that the root is not
            well-approximated by a line (e.g. a cell body.). The density for those vertices
            will be infinite or nan.
            Default is False.


        Returns
        -------
        density_estimate : array
            N-length array of density at all mesh vertices.
        """
        W = window_matrix(self.skeleton, width)

        inds = self._convert_to_meshindex(inds)
        has_inds = np.full(self.skeleton.n_vertices, 0)
        if weight is None:
            skinds, count = np.unique(inds.to_skel_index, return_counts=True)
            has_inds[skinds] = count
        else:
            for w, skind in zip(weight, inds.to_skel_index):
                has_inds[skind] += w
        item_count = W.dot(has_inds.reshape(-1, 1)).ravel()
        if normalize:
            if exclude_root:
                g = self.skeleton.cut_graph(
                    self.skeleton.child_nodes([self.skeleton.root])[0], directed=False
                )
                len_per = np.array(g.sum(axis=1) / 2).ravel()
            else:
                len_per = np.array(
                    self.skeleton.csgraph_undirected.sum(axis=1) / 2
                ).ravel()

            norm = W.dot(len_per.reshape(-1, 1)).ravel()
            with np.errstate(divide="ignore"):
                rho = item_count / norm
        else:
            rho = item_count
        return rho[self.skeleton.mesh_to_skel_map][self.mesh.node_mask]

    ###k########################
    # Visualization functions #
    ###########################

    def requires_vtk(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if _vtk_loaded:
                return func(*args, **kwargs)
            else:
                return None

        return wrapper

    @requires_vtk
    def mesh_actor(self, **kwargs):
        if self.mesh is not None:
            return trimesh_vtk.mesh_actor(self.mesh, **kwargs)
        else:
            return None

    @requires_vtk
    def anno_point_actor(self, anno_name, query=None, filter_query=None, **kwargs):
        row_filter = np.full(len(self.anno[anno_name].df), True)
        if query is not None:
            qry = self.anno[anno_name].query(query)
            row_filter = np.logical_and(row_filter, qry.row_filter)
        if filter_query is not None:
            qry = self.anno[anno_name].filter_query(filter_query)
            row_filter = np.logical_and(row_filter, qry.row_filter)
        if anno_name in self.anno.table_names:
            return trimesh_vtk.point_cloud_actor(
                self.anno[anno_name].points[row_filter], **kwargs
            )

    @OnlyIfSkeleton.exists
    @requires_vtk
    def skeleton_actor(self, **kwargs):
        if self.skeleton is not None:
            return trimesh_vtk.skeleton_actor(self.skeleton, **kwargs)

    @property
    @OnlyIfSkeleton.exists
    def radius(self):
        return self.skeleton_property_to_mesh(self.skeleton.radius)

    ##########
    # Saving #
    ##########
    def save_meshwork(
        self, filename, overwrite=False, version=meshwork_io.LATEST_VERSION
    ):
        """Save meshwork to hdf5 file.

        Parameters
        ----------
        filename : str
            Filename to save to
        overwrite : bool, optional
            If True, overwrites an existing file. Default is False.
        """
        meshwork_io._save_meshwork(filename, self, overwrite=overwrite, version=version)

    @OnlyIfSkeleton.exists
    def export_to_swc(
        self,
        filename,
        resample_spacing=None,
        apical_label=None,
        axon_label=None,
        dendrite_label=None,
        soma_label=None,
        dendrite_default=True,
        interp_kind="linear",
        tip_length_ratio=0.5,
        radius=None,
        radius_agg="mean",
        header=None,
        scaling=1000,
        resample_soma=False,
    ):
        """Export a neuron to SWC, optionally resampling and using annotations to label compartments.

        Parameters
        ----------
        filename : str
            Location where the swc file will be saved
        resample_spacing : float or None, optional
            Desired spacing between skeleton vertices. If None, no resampling is performed. Default is None.
        apical_label : str or array, optional
            If a string, name of annotation spanning the apical dendrite compartment. If a numpy array, one that
            includes all mesh vertices that should be labeled as apical. SkeletonIndex or MeshIndex arrays that
            cover all such apical vertices are also interpreted correctly. By default, None
        axon_label : str or array, optional
            Same as apical_label, but for the axon compartment. by default None.
        dendrite_label : str or array, optional
            Same as apical_label, but for the (basal) dendrite compartment. by default None.
        soma_label : str or array, optional
            Same as apical_label, but for the soma compartment. by default None.
        dendrite_default : bool, optional
            If True, assumes any vertex without a specific label is basal dendrite, by default True.
        interp_kind : str, optional
            Interpolation method used by scipy.interpolate.interp1d, by default "linear".
        tip_length_ratio : float, optional
            Ratio of spacing of branch tip length to desired spacing to keep a final tip, by default 0.5.
        radius : float or array, optional
            Array of radius values for the skeleton (in the same units as vertex coordinates), by default None.
            Array can be either one value per mesh vertex (in which case values are aggregated) or one value per skeleton vertex.
            Alternatively, a single number will be assigned for all vertices.
        radius_agg : str or func, optional
            If mapping mesh radius values to skeleton vertices, how to aggregate values. Options are 'mean', 'median',
            'min', 'max', and a user-defined aggregation function. By default, "mean".
        header : str, optional
            Header string for the SWC file, by default None.
        scaling : float, optional
            Scaling between coordinates of the file and exported coordiates, by default 1000 because for EM
            we usually have coordinates in nanometers and SWC usually expects microns.
        """

        def _extract_annotation(anno_value):
            if anno_value is not None:
                if isinstance(anno_value, str):
                    return self.anno[anno_value].skel_mask
                else:
                    anno_minds = self._convert_to_meshindex(anno_value)
                    return anno_minds.to_skel_mask
            else:
                return None

        apical_inds = _extract_annotation(apical_label)
        axon_inds = _extract_annotation(axon_label)
        dendrite_inds = _extract_annotation(dendrite_label)
        soma_inds = _extract_annotation(soma_label)

        node_labels = swc_node_labels(
            self.skeleton,
            dendrite_indices=dendrite_inds,
            apical_indices=apical_inds,
            soma_indices=soma_inds,
            axon_indices=axon_inds,
            dendrite_default=dendrite_default,
        )

        if radius is not None:
            if np.isscalar(radius):
                radius = np.full(self.skeleton.n_vertices, radius)
            elif len(radius) == self.mesh.n_vertices:
                radius = self.mesh_property_to_skeleton(radius, aggfunc=radius_agg)
            elif len(radius) != self.skeleton.n_vertices:
                raise ValueError(
                    "Radius array must correspond to either mesh or skeleton vertices"
                )

        self.skeleton.export_to_swc(
            filename=filename,
            node_labels=node_labels,
            radius=radius,
            header=header,
            xyz_scaling=scaling,
            resample_spacing=resample_spacing,
            interp_kind=interp_kind,
            tip_length_ratio=tip_length_ratio,
            avoid_root=not resample_soma,
        )


def load_meshwork(filename):
    """Loads meshwork file from an hdf5 file.

    Parameters
    ----------
    filename : str
        file location

    Returns
    -------
    meshwork : Meshwork
        Meshwork object from file
    """
    meta, mesh, skel, annos, mask = meshwork_io._load_meshwork(filename)
    mw = Meshwork(
        mesh,
        skeleton=skel,
        seg_id=meta.get("seg_id", None),
        voxel_resolution=meta.get("voxel_resolution", DEFAULT_VOXEL_RESOLUTION),
    )
    for name, data in annos.items():
        mw.add_annotations(
            name=name,
            data=data.get("data"),
            anchored=data.get("anchor_to_mesh"),
            point_column=data.get("point_column"),
            max_distance=data.get("max_distance"),
            index_column=data.get("index_column", None),
        )
    if not np.all(mask):
        mw.apply_mask(mask)
    return mw
