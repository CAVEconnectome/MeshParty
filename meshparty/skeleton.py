import numpy as np
from meshparty import utils
from scipy import spatial, sparse, interpolate
try:
    from pykdtree.kdtree import KDTree as pyKDTree
except:
    pyKDTree = spatial.cKDTree
from copy import copy
import json
from meshparty import skeleton_io
from collections.abc import Iterable
from meshparty.trimesh_io import Mesh


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
            raise ValueError(
                "New root must correspond to a skeleton vertex index")
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
                comp_root = new_root
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
        """ np.array : N length array with the distance to the root node along the skeleton.
        """
        if self._distance_to_root is None:
            self._distance_to_root = sparse.csgraph.dijkstra(
                self.csgraph, directed=False, indices=self.root
            )
        return self._distance_to_root

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
        if node_mask is not None:
            self.apply_mask(node_mask, in_place=True)

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
        """ numpy.array : N_mesh length array giving the associated skeleton
        vertex for each mesh vertex"""
        if self._rooted.mesh_to_skel_map is None:
            return None
        else:
            return self.filter_unmasked_indices_padded(self._rooted.mesh_to_skel_map)

    @property
    def mesh_to_skel_map_base(self):
        """ numpy.array : N_mesh length array giving the associated skeleton
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
        self._reset_derived_properties_filtered()

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
        """an an array of distances to root for each skeleton vertex
        "Distance to root (even if root is not in the mask)"
        Returns
        -------
        np.array
            N length array with the distance to the root node along the skeleton. 
        """
        
        return self._rooted.distance_to_root[self.node_mask]
    
    def resample(self, spacing, kind='nearest'):
        """ resample the skeleton to a new spacing and filter it to only include components connected to root
        
        Parameters
        ----------
        spacing : [float]
            desired edge spacing in units of vertices

        Returns
        -------
        skeleton.Skeleton
            a resampled skeleton, which has the parts which are not connected to root removed
        resample_map:
            a N long array with as many entries as vertices in the  original skeleton.
            the entry reflects which new skeleton vertex that vertex should be mapped to
        """
        cpaths= self.cover_paths
        d_to_root = self.distance_to_root
        path_counter=0
        branch_d = {}
        vert_list= []
        edge_list = []
    
        resample_map = -np.ones(len(self.vertices), dtype=np.int32)
        
        for path in cpaths:
            if ~np.isinf(d_to_root[path[-1]]):
                # use the distance from root to parameterize the path
                input_d = d_to_root[path]
                # the desired distances from root are evenly spaced according to spacing
                des_d = np.arange(np.min(input_d),np.max(input_d), spacing)
                # setup an interpolation function based upon distance to root as input and xyz as output
                fi = interpolate.interp1d(input_d, self.vertices[path,:], kind=kind, axis=0)
                # use the function to interpolate the new values
                new_verts = fi(des_d)
        
                # find the index of the old branch points in the new path
                is_branch = np.isin(np.array(path), self.branch_points)
                path_branch = path[is_branch]
                path_branch_verts = self.vertices[path_branch, :]
                tree = pyKDTree(new_verts)
                map_ds, new_branch_on_path = tree.query(path_branch_verts)
                new_branch_on_path+=path_counter
                # create a temporary dictionary with this path's branch  points
                new_branch_d = {pb: nw for pb, nw in zip(path_branch, new_branch_on_path)}
                # update the overall mapping dictionary
                branch_d.update(new_branch_d)

                # map the entire path to the new vertices by euc distance
                map_ds, path_map = tree.query(self.vertices[path,:])
                # update the mapping
                resample_map[path]=path_map+path_counter
                
                # new edges just march down path from last vertex to first
                new_edges = np.vstack([ np.arange(len(new_verts)-1,0,-1), np.arange(len(new_verts)-2,-1,-1)]).T + path_counter
                # need to construct the last edge since it wasn't in the original
                # find that last edge whose start point was the first vertex in the path
                last_edge=self.edges[self.edges[:,0]==path[-1],:]
                # for the first path there won't be an edge as the first vertex is root
                if len(last_edge)==1:
                    # if we do have one, then we want to add an edge that is the from the first vertex
                    # in the path, to the new vertex (mapped through branch_d) of the edge we found
                    # in the original edge list
                    new_edges = np.vstack([new_edges, [path_counter, branch_d[last_edge[0,1]] ]])
                # collect the edges and vertices in a list
                edge_list.append(new_edges)
                vert_list.append(new_verts)
                # increment the counter to keep track of how many  vertices we have
                path_counter += len(new_verts)
           
        # concatenate the results together
        new_verts = np.vstack(vert_list)
        new_edges = np.vstack(edge_list)
   
        # create a new skeleton
        # TODO add options to update mesh mapping and vertex properties
        return Skeleton(new_verts, new_edges,
                        root=branch_d[self.root]), resample_map

    def path_to_root(self, v_ind):
        "Path stops if it leaves masked region"
        path_b = self._rooted.path_to_root(self.map_indices_to_unmasked(v_ind))
        path_filt = self.filter_unmasked_indices_padded(path_b)
        if np.any(path_filt == -1):
            last_ind = np.flatnonzero(path_filt == -1)[0]
        else:
            last_ind = len(path_filt)
        return self.SkeletonIndex(path_filt[:last_ind])

    #######################
    # Filtered properties #
    #######################

    def _reset_derived_properties_filtered(self):
        self._vertices = None
        self._edges = None
        self._kdtree = None
        self._pykdtree = None
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
        """Compute the length of a single path (assumed to be correct)
        """
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

        if isinstance(paths[0], Iterable):
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
        """Pre-compute branch and end points from the graph
        """
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
        """End points without skeleton orientation, including root and disconnected components.
        """
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
        segments = [
            self.SkeletonIndex(np.flatnonzero(invs == ii)) for ii in np.unique(invs)
        ]
        segment_map = invs

        return segments, segment_map.astype(int)

    @property
    def segments(self):
        """ list : A list of numpy.array indicies of segments, paths from each branch or
        end point (inclusive) to the next rootward branch/root point (exclusive), that
        cover the skeleton"""
        if self._segments is None:
            self._segments, self._segment_map = self._compute_segments()
        return self._segments

    @property
    def segment_map(self):
        """ np.array : N set of of indices between 0 and len(self.segments)-1, denoting
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
            return self.SkeletonIndex(utils.path_from_predecessors(Ps, t_ind))
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
            d = sparse.csgraph.dijkstra(g.T, indices=[vind])
            if inclusive is False:
                d[vind] = np.inf
            dns.append(self.SkeletonIndex(np.flatnonzero(~np.isinf(d[0]))))

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
            cinds.append(self.SkeletonIndex(
                self.edges[self.edges[:, 1] == vind, 0]))

        if return_single:
            cinds = cinds[0]
        return cinds

    #####################
    # Cover Path Functions #
    #####################

    def _compute_cover_paths(self):
        """Compute the list of cover paths along the skeleton
        """
        cover_paths = []
        seen = np.full(self.n_vertices, False)
        ep_order = np.argsort(self.distance_to_root[self.end_points])[::-1]
        for ep in self.end_points[ep_order]:
            ptr = np.array(self.path_to_root(ep))
            cover_paths.append(self.SkeletonIndex(ptr[~seen[ptr]]))
            seen[ptr] = True
        return cover_paths

    @property
    def cover_paths(self):
        """ list : List of numpy.array objects with self.n_end_points elements, each a rootward
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
        self, filename, node_labels=None, radius=None, header=None, xyz_scaling=1000
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
        )
