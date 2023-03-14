import numpy as np
from meshparty import utils
from scipy import spatial, sparse
from dataclasses import dataclass, fields, asdict, make_dataclass

try:
    from pykdtree.kdtree import KDTree as pyKDTree
except:
    pyKDTree = spatial.cKDTree
from meshparty import skeleton_io
from collections.abc import Iterable
from .skeleton_utils import resample_path


def _metadata_from_dict(
    meta_dict,
    dataclass_name="MetaMetadata",
):
    meta = make_dataclass(dataclass_name, fields=meta_dict.keys())
    return meta(**meta_dict)


@dataclass
class SkeletonMetadata:
    root_id: int = None
    soma_pt_x: float = None
    soma_pt_y: float = None
    soma_pt_z: float = None
    soma_radius: float = None
    collapse_soma: bool = None
    collapse_function: str = None
    invalidation_d: float = None
    smooth_vertices: bool = None
    compute_radius: bool = None
    shape_function: str = None
    smooth_iterations: int = None
    smooth_neighborhood: int = None
    smooth_r: float = None
    cc_vertex_thresh: int = None
    remove_zero_length_edges: bool = None
    collapse_params: dict = None
    timestamp: float = None
    skeleton_type: str = None
    meta: object = None

    # Fields used for skeletonization
    _skeletonize_fields = [
        "soma_pt",
        "soma_radius",
        "collapse_soma",
        "collapse_function",
        "invalidation_d",
        "smooth_vertices",
        "compute_radius",
        "shape_function",
        "smooth_iterations",
        "smooth_neighborhood",
        "smooth_r",
        "cc_vertex_thresh",
        "remove_zero_length_edges",
        "collapse_params",
    ]

    def __init__(self, **kwargs):
        names = [f.name for f in fields(self)]

        if kwargs.get("meta") is not None:
            setattr(
                self, "meta", _metadata_from_dict(kwargs.pop("meta"), "MetaMetadata")
            )

        for k, v in kwargs.items():
            if k in names:
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                setattr(self, k, v)

    def skeletonize_kwargs(self):
        params = asdict(self)

        # reassemble soma point into list
        soma_pt = [
            params.pop("soma_pt_x"),
            params.pop("soma_pt_y"),
            params.pop("soma_pt_z"),
        ]
        if soma_pt[0] is not None:
            params["soma_pt"] = soma_pt
        else:
            params["soma_pt"] = None

        for k in list(params.keys()):
            if k not in self._skeletonize_fields:
                params.pop(k)
        return params

    def update_metameta(self, metameta):
        if self.meta is not None:
            meta_dict = asdict(self.meta)
        else:
            meta_dict = {}

        meta_dict.update(metameta)
        setattr(self, "meta", _metadata_from_dict(meta_dict, "MetaMetadata"))
        pass


class StaticSkeleton:
    def __init__(
        self,
        vertices,
        edges,
        root=None,
        radius=None,
        mesh_to_skel_map=None,
        mesh_index=None,
        vertex_properties=None,
        voxel_scaling=None,
    ):
        self._vertices = vertices
        self._edges = edges
        self._root = None
        self._radius = radius
        self._mesh_to_skel_map = mesh_to_skel_map
        self._mesh_index = mesh_index

        self._parent_node_array = None
        self._distance_to_root = None
        self._hops_to_root = None
        self._csgraph = None
        self._csgraph_binary = None
        self._voxel_scaling = voxel_scaling

        if root is None:
            self._create_default_root()
        else:
            self.reroot(root, reset_other_components=True)

        self._reset_derived_properties()
        self.vertex_properties = vertex_properties

    @property
    def vertices(self):
        return self._vertices

    @property
    def edges(self):
        return self._edges

    @property
    def mesh_to_skel_map(self):
        return self._mesh_to_skel_map

    @property
    def voxel_scaling(self):
        if self._voxel_scaling is None:
            return None
        else:
            return np.array(self._voxel_scaling)

    @voxel_scaling.setter
    def voxel_scaling(self, new_scaling):
        self._vertices = self._vertices * self.inverse_voxel_scaling
        if new_scaling is not None:
            self._vertices = self._vertices * np.array(new_scaling).reshape(3)
        self._voxel_scaling = new_scaling
        self._reset_derived_properties()

    @property
    def inverse_voxel_scaling(self):
        if self.voxel_scaling is None:
            return np.array([1, 1, 1])
        else:
            return 1 / self.voxel_scaling

    @property
    def n_vertices(self):
        """ int : Number of vertices in the skeleton """
        return len(self.vertices)

    @property
    def root(self):
        """ int : Index of the skeleton root """
        if self._root is None:
            self._create_default_root()
        return self._root

    @property
    def radius(self):
        if self._radius is None:
            return None
        else:
            return self._radius

    @radius.setter
    def radius(self, new_values):
        if len(new_values) == self.n_vertices:
            self._radius = np.array(new_values).reshape(self.n_vertices)

    @property
    def mesh_index(self):
        return self._mesh_index

    def _create_default_root(self):
        temp_graph = utils.create_csgraph(
            self.vertices, self.edges, euclidean_weight=True, directed=False
        )
        r = utils.find_far_points_graph(temp_graph)
        self.reroot(int(r[0]), reset_other_components=True)

    def reroot(self, new_root, reset_other_components=False):
        """Change the skeleton root index.

        Parameters
        ----------
        new_root : Int
            Skeleton vertex index to be the new root.

        reset_other_components : Bool
            Orders non-root components accoring to a local default "root".
            Should not often be set to True by a user.
        """
        if new_root > self.n_vertices:
            raise ValueError("New root must correspond to a skeleton vertex index")
        self._root = int(new_root)
        self._parent_node_array = np.full(self.n_vertices, None)

        _, lbls = sparse.csgraph.connected_components(self.csgraph_binary)
        root_comp = lbls[new_root]
        if reset_other_components:
            comps_to_reroot = np.unique(lbls)
        else:
            comps_to_reroot = [root_comp]
        # The edge list has to be treated like an undirected graph

        for comp in comps_to_reroot:
            if comp == root_comp:
                comp_root = int(new_root)
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
                np.isin(self.edges, np.flatnonzero(lbls == comp)), axis=1
            )

            edges = self.edges[edge_slice]
            is_ordered = d[edges[:, 0]] > d[edges[:, 1]]
            e1 = np.where(is_ordered, edges[:, 0], edges[:, 1])
            e2 = np.where(is_ordered, edges[:, 1], edges[:, 0])
            self._edges[edge_slice] = np.stack((e1, e2)).T
            self._parent_node_array[e1] = e2

        self._reset_derived_properties()

    ######################
    # Derived properties #
    ######################

    def _reset_derived_properties(self):
        self._csgraph = None
        self._csgraph_binary = None
        self._distance_to_root = None
        self._hops_to_root = None

    @property
    def csgraph(self):
        if self._csgraph is None:
            self._csgraph = utils.create_csgraph(
                self.vertices, self.edges, euclidean_weight=True, directed=True
            )
        return self._csgraph

    @property
    def csgraph_binary(self):
        if self._csgraph_binary is None:
            self._csgraph_binary = utils.create_csgraph(
                self.vertices, self.edges, euclidean_weight=False, directed=True
            )
        return self._csgraph_binary

    @property
    def csgraph_undirected(self):
        return self.csgraph + self.csgraph.T

    @property
    def csgraph_binary_undirected(self):
        return self.csgraph_binary + self.csgraph_binary.T

    def parent_nodes(self, vinds):
        """Get a list of parent nodes for specified vertices

        Parameters
        ----------
        vinds : Collection of ints
            Collection of vertex indices

        Returns
        -------
        numpy.array
            The parent node of each vertex index in vinds.
        """
        if isinstance(vinds, list):
            vinds = np.array(vinds)
        return self._parent_node_array[vinds]

    @property
    def distance_to_root(self):
        """np.array : N length array with the distance to the root node along the skeleton."""
        if self._distance_to_root is None:
            self._distance_to_root = sparse.csgraph.dijkstra(
                self.csgraph, directed=False, indices=self.root
            )
        return self._distance_to_root

    @property
    def hops_to_root(self):
        """np.array : N length array with the number of hops to the root node along the skeleton."""
        if self._hops_to_root is None:
            self._hops_to_root = sparse.csgraph.dijkstra(
                self.csgraph_binary, directed=False, indices=self.root
            )
        return self._hops_to_root

    def path_to_root(self, v_ind):
        """
        Gives the path to root from a specified vertex.

        Parameters
        ----------
        v_ind : int
            Vertex index

        Returns
        -------
        numpy.array : Ordered set of indices from v_ind to root, inclusive of both.
        """
        path = [v_ind]
        ind = v_ind
        ind = self._parent_node_array[ind]
        while ind is not None:
            path.append(ind)
            ind = self._parent_node_array[ind]
        return np.array(path)


class Skeleton:
    def __init__(
        self,
        vertices,
        edges,
        root=None,
        radius=None,
        mesh_to_skel_map=None,
        mesh_index=None,
        vertex_properties={},
        node_mask=None,
        voxel_scaling=None,
        remove_zero_length_edges=True,
        skeleton_index=None,
        meta={},
    ):

        if remove_zero_length_edges:
            zlsk = utils.collapse_zero_length_edges(
                vertices,
                edges,
                root,
                radius,
                mesh_to_skel_map,
                mesh_index,
                node_mask,
                vertex_properties,
            )
            (
                vertices,
                edges,
                root,
                radius,
                mesh_to_skel_map,
                mesh_index,
                node_mask,
                vertex_properties,
            ) = zlsk

        self._rooted = StaticSkeleton(
            vertices,
            edges,
            radius=radius,
            mesh_to_skel_map=mesh_to_skel_map,
            mesh_index=mesh_index,
            vertex_properties=vertex_properties,
            root=root,
            voxel_scaling=voxel_scaling,
        )

        self._node_mask = np.full(self._rooted.n_vertices, True)
        self._edges = None
        self._SkeletonIndex = skeleton_index

        # Derived properties of the filtered graph
        self._csgraph_filtered = None
        self._cover_paths = None
        self._segments = None
        self._segment_map = None
        self._kdtree = None
        self._pykdtree = None
        self._reset_derived_properties_filtered()
        self.vertex_properties = vertex_properties

        if isinstance(meta, SkeletonMetadata):
            self._meta = meta
        else:
            self._meta = SkeletonMetadata(**meta)
        if node_mask is not None:
            self.apply_mask(node_mask, in_place=True)

    @property
    def meta(self):
        return self._meta

    ###################
    # Mask properties #
    ###################

    @property
    def SkeletonIndex(self):
        if self._SkeletonIndex is None:
            self._SkeletonIndex = np.array
        return self._SkeletonIndex

    def _register_skeleton_index(self, NewSkeletonIndex):
        self._SkeletonIndex = NewSkeletonIndex

    @property
    def node_mask(self):
        return self._node_mask

    def copy(self):
        return Skeleton(
            self._rooted.vertices,
            self._rooted.edges,
            mesh_to_skel_map=self._rooted.mesh_to_skel_map,
            vertex_properties=self._rooted.vertex_properties,
            root=self._rooted.root,
            node_mask=self.node_mask,
            radius=self._rooted.radius,
            voxel_scaling=self.voxel_scaling,
            skeleton_index=self._SkeletonIndex,
            mesh_index=self._rooted.mesh_index,
            remove_zero_length_edges=False,
            meta=self.meta,
        )

    def apply_mask(self, new_mask, in_place=False):
        if in_place:
            sk = self
        else:
            sk = self.copy()

        if len(new_mask) == len(sk.vertices):
            all_false = np.full(len(sk.node_mask), False)
            all_false[sk.node_mask] = new_mask
            new_mask = all_false

        sk._node_mask = new_mask
        sk._reset_derived_properties_filtered()

        if in_place is False:
            return sk

    def reset_mask(self, in_place=False):
        true_mask = np.full(self.unmasked_size, True)
        out = self.apply_mask(true_mask, in_place=in_place)
        if in_place is False:
            return out

    def mask_from_indices(self, mask_indices):
        new_mask = np.full(self._rooted.n_vertices, False)
        new_mask[self.map_indices_to_unmasked(mask_indices)] = True
        return new_mask

    @property
    def indices_unmasked(self):
        """
        np.array: Gets the indices of nodes in the filtered mesh in the unmasked index array
        """
        return np.flatnonzero(self.node_mask)

    @property
    def unmasked_size(self):
        return len(self._rooted.vertices)

    def map_indices_to_unmasked(self, unmapped_indices):
        """
        For a set of masked indices, returns the corresponding unmasked indices

        Parameters
        ----------
        unmapped_indices: np.array
            a set of indices in the masked index space

        Returns
        -------
        np.array
            the indices mapped back to the original mesh index space
        """
        return utils.map_indices_to_unmasked(self.indices_unmasked, unmapped_indices)

    def map_boolean_to_unmasked(self, unmapped_boolean):
        """
        For a boolean index in the masked indices, returns the corresponding unmasked boolean index

        Parameters
        ----------
        unmapped_boolean : np.array
            a bool array in the masked index space

        Returns
        -------
        np.array
            a bool array in the original index space.  Is True if the unmapped_boolean suggests it should be.
        """
        return utils.map_boolean_to_unmasked(
            self.unmasked_size, self.node_mask, unmapped_boolean
        )

    def filter_unmasked_boolean(self, unmasked_boolean):
        """
        For an unmasked boolean slice, returns a boolean slice filtered to the masked mesh

        Parameters
        ----------
        unmasked_boolean : np.array
            a bool array in the original mesh index space

        Returns
        -------
        np.array
            returns the elements of unmasked_boolean that are still relevant in the masked index space
        """
        return utils.filter_unmasked_boolean(self.node_mask, unmasked_boolean)

    def filter_unmasked_indices(self, unmasked_shape, mask=None):
        """
        filters a set of indices in the original mesh space
        and returns it in the masked space

        Parameters
        ----------
        unmasked_shape: np.array
            a set of indices into vertices in the unmasked index space
        mask: np.array or None
            the mask to apply. default None will use this Mesh node_mask

        Returns
        -------
        np.array
            the unmasked_shape indices mapped into the masked index space
        """
        if mask is None:
            mask = self.node_mask
        return utils.filter_unmasked_indices(mask, unmasked_shape)

    def filter_unmasked_indices_padded(self, unmasked_shape, mask=None):
        """
        filters a set of indices in the original mesh space
        and returns it in the masked space

        Parameters
        ----------
        unmasked_shape: np.array
            a set of indices into vertices in the unmasked index space
        mask: np.array or None
            the mask to apply. default None will use this Mesh node_mask

        Returns
        -------
        np.array
            the unmasked_shape indices mapped into the masked index space,
            with -1 where the original index did not map into the masked mesh.
        """
        if mask is None:
            mask = self.node_mask
        return utils.filter_unmasked_indices_padded(mask, unmasked_shape)

    ####################
    # Basic properties #
    ####################

    @property
    def vertices(self):
        if self._vertices is None:
            self._vertices = self._rooted.vertices[self.node_mask]
        return self._vertices

    @vertices.setter
    def vertices(self, new_vertices):
        new_vertices = np.atleast_2d(new_vertices)
        if new_vertices.shape[1] != 3:
            raise ValueError("New vertices must be 3 dimensional")
        if len(new_vertices) == self._rooted.n_vertices:
            self._rooted._vertices = new_vertices
        elif len(new_vertices) == self.n_vertices:
            self._rooted._vertices[self.node_mask] = new_vertices
        else:
            raise ValueError("New vertices must be the same size as existing vertices")
        self._reset_derived_properties_rooted()
        self._reset_derived_properties_filtered(index_changed=False)

    @property
    def edges(self):
        if self._edges is None:
            self._edges = self.filter_unmasked_indices(self._rooted.edges)
        return self._edges

    @property
    def n_vertices(self):
        """ int : Number of vertices in the skeleton """
        return len(self.vertices)

    @property
    def mesh_to_skel_map(self):
        """numpy.array : N_mesh length array giving the associated skeleton
        vertex for each mesh vertex"""
        if self._rooted.mesh_to_skel_map is None:
            return None
        else:
            return self.filter_unmasked_indices_padded(self._rooted.mesh_to_skel_map)

    @property
    def mesh_to_skel_map_base(self):
        """numpy.array : N_mesh length array giving the associated skeleton
        vertex for each mesh vertex"""
        if self._rooted.mesh_to_skel_map is None:
            return None
        else:
            return self._rooted.mesh_to_skel_map

    @property
    def radius(self):
        if self._rooted.radius is None:
            return None
        return self._rooted.radius[self.node_mask]

    @property
    def mesh_index(self):
        if self._rooted.mesh_index is None:
            return None
        return self._rooted.mesh_index[self.node_mask]

    @property
    def csgraph(self):
        return self._rooted.csgraph[:, self.node_mask][self.node_mask]

    @property
    def csgraph_binary(self):
        return self._rooted.csgraph_binary[:, self.node_mask][self.node_mask]

    @property
    def csgraph_undirected(self):
        return self._rooted.csgraph_undirected[:, self.node_mask][self.node_mask]

    @property
    def csgraph_binary_undirected(self):
        return self._rooted.csgraph_binary_undirected[:, self.node_mask][self.node_mask]

    ##################
    # Voxel scalings #
    ##################

    @property
    def voxel_scaling(self):
        return self._rooted.voxel_scaling

    @voxel_scaling.setter
    def voxel_scaling(self, new_scaling):
        self._rooted.voxel_scaling = new_scaling
        self._reset_derived_properties_filtered(index_changed=False)

    #####################
    # Rooted properties #
    #####################

    def _reset_derived_properties_rooted(self):
        self._rooted._reset_derived_properties()

    def _create_default_root(self):
        temp_graph = utils.create_csgraph(
            self._rooted.vertices,
            self._rooted.edges,
            euclidean_weight=True,
            directed=False,
        )
        r = utils.find_far_points_graph(temp_graph)
        self._rooted.reroot(int(r[0]))

    @property
    def root(self):
        return self.SkeletonIndex(
            self.filter_unmasked_indices_padded(self._rooted.root)
        )

    @property
    def root_position(self):
        return self._rooted.vertices[self._rooted.root]

    def reroot(self, new_root):
        self._rooted.reroot(self.map_indices_to_unmasked(new_root))
        self._reset_derived_properties_filtered()

    @property
    def distance_to_root(self):
        "Distance to root (even if root is not in the mask)"
        return self._rooted.distance_to_root[self.node_mask]

    def path_to_root(self, v_ind):
        "Path stops if it leaves masked region"
        path_b = self._rooted.path_to_root(self.map_indices_to_unmasked(v_ind))
        path_filt = self.filter_unmasked_indices_padded(path_b)
        if np.any(path_filt == -1):
            last_ind = np.flatnonzero(path_filt == -1)[0]
        else:
            last_ind = len(path_filt)
        return self.SkeletonIndex(path_filt[:last_ind])

    @property
    def hops_to_root(self):
        return self._rooted.hops_to_root[self.node_mask]

    #######################
    # Filtered properties #
    #######################

    def _reset_derived_properties_filtered(self, index_changed=True):
        self._vertices = None
        self._edges = None
        self._kdtree = None
        self._pykdtree = None

        if index_changed:
            self._branch_points = None
            self._end_points = None
            self._segments = None
            self._segment_map = None
            self._SkeletonIndex = None
            self._cover_paths = None

    #########################
    # Geometric quantitites #
    #########################

    @property
    def kdtree(self):
        """ scipy.spatial.kdtree : k-D tree from scipy.spatial. """
        if self._kdtree is None:
            self._kdtree = spatial.cKDTree(self.vertices)
        return self._kdtree

    @property
    def pykdtree(self):
        if self._pykdtree is None:
            self._pykdtree = pyKDTree(self.vertices)
        return self._pykdtree

    def _single_path_length(self, path):
        """Compute the length of a single path (assumed to be correct)"""
        path = np.unique(path)
        return np.sum(self.csgraph[:, path][path])

    def path_length(self, paths=None):
        """Returns the length of a path (described as an ordered collection of connected indices)
        Parameters
        ----------
        path : Path or collection of paths, a path being an ordered list of linked vertex indices.

        Returns
        -------
        Float or list of floats : The length of each path.
        """
        if paths is None:
            paths = np.arange(self.n_vertices)

        if len(paths) == 0:
            return 0

        if np.ndim(paths[0]) > 0:
            Ls = []
            for path in paths:
                Ls.append(self._single_path_length(path))
        else:
            Ls = self._single_path_length(paths)
        return Ls

    ################################
    # Topological split properties #
    ################################

    def _create_branch_and_end_points(self):
        """Pre-compute branch and end points from the graph"""
        n_children = np.sum(self.csgraph_binary > 0, axis=0).squeeze()
        self._branch_points = np.flatnonzero(n_children > 1)
        self._end_points = np.flatnonzero(n_children == 0)

    @property
    def branch_points(self):
        """ numpy.array : Indices of branch points on the skeleton (pottentially including root)"""
        if self._branch_points is None:
            self._create_branch_and_end_points()
        return self.SkeletonIndex(self._branch_points)

    @property
    def end_points(self):
        """ numpy.array : Indices of end points on the skeleton (pottentially including root)"""
        if self._end_points is None:
            self._create_branch_and_end_points()
        return self.SkeletonIndex(self._end_points)

    @property
    def topo_points(self):
        return self.SkeletonIndex(
            np.concatenate([self.end_points, self.branch_points, [self.root]])
        )

    @property
    def n_branch_points(self):
        return len(self.branch_points)

    @property
    def n_end_points(self):
        return len(self.end_points)

    @property
    def end_points_undirected(self):
        """End points without skeleton orientation, including root and disconnected components."""
        return self.SkeletonIndex(
            np.flatnonzero(np.sum(self.csgraph_binary_undirected, axis=0) == 1)
        )

    @property
    def branch_points_undirected(self):
        return self.SkeletonIndex(
            np.flatnonzero(np.sum(self.csgraph_binary_undirected, axis=0) > 2)
        )

    def _compute_segments(self):
        """Precompute segments between branches and end points"""
        segments = []
        segment_map = np.zeros(len(self.vertices)) - 1

        if len(self.branch_points) > 0:
            ch_inds = np.concatenate(self.child_nodes(self.branch_points))
            g = self.cut_graph(ch_inds)
            _, ls = sparse.csgraph.connected_components(g)
        else:
            _, ls = sparse.csgraph.connected_components(self.csgraph_binary)

        _, invs = np.unique(ls, return_inverse=True)
        for ii in np.unique(invs):
            seg = self.SkeletonIndex(np.flatnonzero(invs == ii))
            segments.append(
                seg[np.argsort(self.hops_to_root[seg])[::-1]]
            )
        segment_map = invs

        return segments, segment_map.astype(int)

    @property
    def segments(self):
        """list : A list of numpy.array indicies of segments, paths from each branch or
        end point (inclusive) to the next rootward branch/root point (exclusive), that
        cover the skeleton"""
        if self._segments is None:
            self._segments, self._segment_map = self._compute_segments()
        return self._segments

    @property
    def segments_plus(self):
        """list : A list of array indices of segments, including the segment parent in the next segmetn"""
        segs_plus = []
        for seg in self.segments:
            parent = self.parent_nodes(seg[-1])
            if parent >= 0:
                segs_plus.append(self.SkeletonIndex(np.concatenate([seg, [parent]])))
            else:
                segs_plus.append(seg)
        return segs_plus

    @property
    def segment_map(self):
        """np.array : N set of of indices between 0 and len(self.segments)-1, denoting
        which segment a given skeleton vertex is in.
        """
        if self._segment_map is None:
            self._segments, self._segment_map = self._compute_segments()
        return self._segment_map

    def path_between(self, s_ind, t_ind):
        d, Ps = sparse.csgraph.dijkstra(
            self.csgraph_binary_undirected,
            directed=False,
            indices=s_ind,
            return_predecessors=True,
        )
        if not np.isinf(d[t_ind]):
            return self.SkeletonIndex(utils.path_from_predecessors(Ps, t_ind)[::-1])
        else:
            return None

    ############################
    # Relative node properties #
    ############################

    def parent_nodes(self, vinds):
        """Get a list of parent nodes for specified vertices

        Parameters
        ----------
        vinds : Collection of ints
            Collection of vertex indices

        Returns
        -------
        numpy.array
            The parent node of each vertex index in vinds.
        """
        vinds_b = self.map_indices_to_unmasked(vinds)
        if isinstance(vinds_b, int):
            vinds_b = np.array([vinds_b])
        p_inds_filtered = self.filter_unmasked_indices_padded(
            self._rooted.parent_nodes(vinds_b)
        )
        return self.SkeletonIndex(p_inds_filtered)

    def cut_graph(self, vinds, directed=True, euclidean_weight=True):
        """Return a csgraph for the skeleton with specified vertices cut off from their parent vertex.

        Parameters
        ----------
        vinds :
            Collection of indices to cut off from their parent.
        directed : bool, optional
            Return the graph as directed, by default True
        euclidean_weight : bool, optional
            Return the graph with euclidean weights, by default True. If false, the unweighted.

        Returns
        -------
        scipy.sparse.csr.csr_matrix
            Graph with vertices in vinds cutt off from parents.
        """
        e_keep = ~np.isin(self.edges[:, 0], vinds)
        es_new = self.edges[e_keep]
        return utils.create_csgraph(
            self.vertices, es_new, euclidean_weight=euclidean_weight, directed=directed
        )

    def downstream_nodes(self, vinds, inclusive=True):
        """Get a list of all nodes downstream of a collection of indices

        Parameters
        ----------
        vinds : Collection of ints
            Collection of vertex indices

        Returns
        -------
        List of arrays
            List whose ith element is an array of all vertices downstream of the ith element of vinds.
        """
        vinds, return_single = utils.array_if_scalar(vinds)

        dns = []
        for vind in vinds:
            g = self.cut_graph(vind)
            d = sparse.csgraph.dijkstra(g.T, indices=[vind]).squeeze()
            if inclusive is False:
                d[vind] = np.inf
            dns.append(self.SkeletonIndex(np.flatnonzero(~np.isinf(d))))

        if return_single:
            dns = dns[0]
        return dns

    def child_nodes(self, vinds):
        """Get a list of all immediate children of list of indices

        Parameters
        ----------
        vinds : Collection of ints
            Collection of vertex indices

        Returns
        -------
        List of arrays
            A list whose ith element is the children of the ith element of vinds.
        """
        vinds, return_single = utils.array_if_scalar(vinds)
        # return_single = False
        # if np.isscalar(vinds):
        #     vinds = [vinds]
        #     return_single = True
        # elif issubclass(type(vinds), np.ndarray):
        #     if len(vinds.shape) == 0:
        #         vinds = vinds.reshape(1)
        #         return_single = True

        cinds = []
        for vind in vinds:
            cinds.append(self.SkeletonIndex(self.edges[self.edges[:, 1] == vind, 0]))

        if return_single:
            cinds = cinds[0]
        return cinds

    #####################
    # Cover Path Functions #
    #####################

    def _compute_cover_paths(self, end_points=None, include_parent=False):
        """Compute the list of cover paths along the skeleton"""
        cover_paths = []
        seen = np.full(self.n_vertices, False)
        if end_points is None:
            end_points = self.end_points

        ep_order = np.argsort(self.distance_to_root[end_points])[::-1]
        for ep in end_points[ep_order]:
            ptr = np.array(self.path_to_root(ep))
            path = ptr[~seen[ptr]]
            if include_parent:
                pn = int(self.parent_nodes(path[-1]))
                if pn != -1:
                    path = np.concatenate((path, [pn]))
            cover_paths.append(self.SkeletonIndex(path))
            seen[ptr] = True
        return cover_paths

    @property
    def cover_paths(self):
        """list : List of numpy.array objects with self.n_end_points elements, each a rootward
        path (ordered set of indices) starting from an endpoint and continuing until it reaches
        a point on a path earlier on the list. Paths are ordered by end point distance from root,
        starting with the most distal. When traversed from begining to end, gives the longest rootward
        path down each major branch first. When traversed from end to begining, guarentees every
        branch point is visited at the end of all its downstream paths before being traversed along
        a path.
        """
        if self._cover_paths is None:
            self._cover_paths = self._compute_cover_paths()
        return self._cover_paths

    def cover_paths_specific(self, end_points, include_parent=False):
        """Compute nonoverlapping paths from specified endpoints

        Parameters
        ----------
            end_points : array-like
                Array of skeleton vertices to use as end points

        Returns
        -------
            array
                List of cover paths using the specified end points. Note that this is not sorted in the same order
                (or necessarily the same length) as specified end points.
        """
        paths = self._compute_cover_paths(
            end_points=end_points, include_parent=include_parent
        )
        return [p for p in paths if len(p) > 0]

    def cover_paths_with_parent(self):
        """Compute minimally overlapping paths toward root, including a single parent vertex where the path originates."""
        return self._compute_cover_paths(include_parent=True)

    ####################
    # Export functions #
    ####################

    def write_to_h5(self, filename, overwrite=True):
        """Write the skeleton to an HDF5 file. Note that this is done in the original dimensons, not the scaled dimensions.

        Parameters
        ----------
        filename : str
            Filename to save file
        overwrite : bool, optional
            Flag to specify whether to overwrite existing files, by default True
        """
        existing_voxel_scaling = self.voxel_scaling
        self.voxel_scaling = None
        skeleton_io.write_skeleton_h5(self, filename, overwrite=overwrite)
        self.voxel_scaling = existing_voxel_scaling

    def export_to_swc(
        self,
        filename,
        node_labels=None,
        radius=None,
        header=None,
        xyz_scaling=1000,
        resample_spacing=None,
        interp_kind="linear",
        tip_length_ratio=0.5,
        avoid_root=True,
    ):
        """
        Export a skeleton file to an swc file
        (see http://research.mssm.edu/cnic/swc.html for swc definition)

        Parameters
        ----------
        filename : str
            Name of the file to save the swc to
        node_labels : np.array
            None (default) or an interable of ints co-indexed with vertices.
            Corresponds to the swc node categories. Defaults to setting all
            odes to label 3, dendrite.
        radius : iterable
            None (default) or an iterable of floats. This should be co-indexed with vertices.
            Radius values are assumed to be in the same units as the node vertices.
        header : dict
            default None. Each key value pair in the dict becomes
            a parameter line in the swc header.
        xyz_scaling : Number
            default 1000. Down-scales spatial units from the skeleton's units to
            whatever is desired by the swc. E.g. nm to microns has scaling=1000.

        """
        skeleton_io.export_to_swc(
            self,
            filename,
            node_labels=node_labels,
            radius=radius,
            header=header,
            xyz_scaling=xyz_scaling,
            resample_spacing=resample_spacing,
            interp_kind=interp_kind,
            tip_length_ratio=tip_length_ratio,
            avoid_root=avoid_root,
        )


def resample(sk, spacing, kind="linear", tip_length_ratio=0.5, avoid_root=True):
    """Resample a skeleton's vertices

    Parameters
    ----------
    sk : Skeleton
        Input skeleton file with a skeleton
    spacing : numeric
        Desired spacing in nanometers
    kind : str, optional
        Type of interpolation to use when resampling. Options follow scipy.interpolate.interp1d. By default "linear"
    tip_length_ratio : float, optional
        The ratio of spacing to branch tip length that a branch tip must have in order to be included in the final skeleton
        for example: spacing is 10 and branch length is 8. do you want to include that final 8 length tip?
        then perhaps consider a tip_length_ratio of .75, by default 0.25

    Returns
    -------
    Skeleton
        New skeleton with resampled vertices.

    resample_map
        Array where the ith index corresponds to the ith vertex of the resampled skeleton and the value
        is the associated index in the original skeleton. To assign vertices, we assign a "domain" to each
        vertex in the original skeleton that is halfway between the vertex and its neighbors. Resampled
        vertices that fall within that domain (based on topology and distance-to-root) are then associated
        with the original vertex.
    """
    path_counter = 0
    branch_d = {}
    vert_list = []
    edge_list = []
    output_map_list = []

    for path in sk.cover_paths:
        new_verts, new_edges, output_map_path, branch_d = resample_path(
            path,
            sk,
            path_counter,
            spacing,
            kind,
            tip_length_ratio,
            branch_d,
            avoid_root,
        )
        vert_list.append(new_verts)
        edge_list.append(new_edges)
        output_map_list.append(output_map_path)
        path_counter += len(new_verts)

    new_verts = np.vstack(vert_list)
    new_edges = np.vstack(edge_list)
    resample_map = np.concatenate(output_map_list)

    return (
        Skeleton(
            new_verts,
            new_edges,
            root=branch_d[int(sk.root)],
            remove_zero_length_edges=False,
        ),
        resample_map,
    )