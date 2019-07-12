import numpy as np
import h5py
from scipy import spatial, sparse
from sklearn import decomposition
from pykdtree.kdtree import KDTree
import os
import networkx as nx
import requests
import time
from collections import defaultdict
import warnings

import cloudvolume
from multiwrapper import multiprocessing_utils as mu

import trimesh
from trimesh import caching
try:
    from trimesh import exchange
except ImportError:
    from trimesh import io as exchange

from pymeshfix import _meshfix
from tqdm import trange

from meshparty import utils, trimesh_repair

def read_mesh_h5(filename):
    """Reads a mesh's vertices, faces and normals from an hdf5 file"""
    assert os.path.isfile(filename)

    with h5py.File(filename, "r") as f:
        vertices = f["vertices"][()]
        faces = f["faces"][()]

        if len(faces.shape) == 1:
            faces = faces.reshape(-1, 3)

        if "normals" in f.keys():
            normals = f["normals"][()]
        else:
            normals = []

        if "link_edges" in f.keys():
            link_edges = f["link_edges"][()]
        else:
            link_edges = None
        
        if "node_mask" in f.keys():
            node_mask = f["node_mask"][()]
        else:
            node_mask = None
    return vertices, faces, normals, link_edges, node_mask


def write_mesh_h5(filename, vertices, faces,
                  normals=None, link_edges=None, node_mask=None, overwrite=False):
    """Writes a mesh's vertices, faces (and normals) to an hdf5 file"""

    if os.path.isfile(filename):
        if overwrite:
            os.remove(filename)
        else:
            return

    with h5py.File(filename, "w") as f:
        f.create_dataset("vertices", data=vertices, compression="gzip")
        f.create_dataset("faces", data=faces, compression="gzip")

        if normals is not None:
            f.create_dataset("normals", data=normals, compression="gzip")

        if link_edges is not None:
            f.create_dataset("link_edges", data=link_edges, compression="gzip")

        if node_mask is not None:
            f.create_dataset("node_mask", data=node_mask, compression="gzip")


def read_mesh(filename):
    """Reads a mesh's vertices, faces and normals from obj or h5 file"""

    if filename.endswith(".obj"):
        with open(filename,'r') as fp:
            mesh_d = exchange.wavefront.load_wavefront(fp)
        vertices = mesh_d[0]['vertices']
        faces =  mesh_d[0]['faces']
        normals = mesh_d[0].get('normals', None)
        link_edges = None
        node_mask = None
    elif filename.endswith(".h5"):
        mesh_data = read_mesh_h5(filename)
        vertices, faces, normals, link_edges, node_mask = mesh_data
    else:
        raise Exception("Unknown filetype")
    return vertices, faces, normals, link_edges, node_mask


def _download_meshes_thread(args):
    """ Helper to Download meshes into target directory """
    seg_ids, cv_path, target_dir, fmt, overwrite, \
        merge_large_components, stitch_mesh_chunks, map_gs_to_https = args

    cv = cloudvolume.CloudVolumeFactory(cv_path, map_gs_to_https=map_gs_to_https)

    for seg_id in seg_ids:
        print('downloading {}'.format(seg_id))
        target_file = os.path.join(target_dir, f"{seg_id}.h5")
        if not overwrite and os.path.exists(target_file):
            print('file exists {}'.format(target_file))
            continue
        print('file does not exist {}'.format(target_file))

        try:
            cv_mesh = cv.mesh.get(seg_id, remove_duplicate_vertices=False)

            faces = np.array(cv_mesh["faces"])
            if len(faces.shape) == 1:
                faces = faces.reshape(-1, 3)

            mesh = Mesh(vertices=cv_mesh["vertices"],
                        faces=faces,
                        process=False)

            if merge_large_components:
                mesh.merge_large_components()

            if fmt == "hdf5":
                write_mesh_h5(f"{target_dir}/{seg_id}.h5",
                              mesh.vertices,
                              mesh.faces.flatten(),
                              link_edges=mesh.link_edges,
                              overwrite=overwrite)
            else:
                mesh.write_to_file(f"{target_dir}/{seg_id}.{fmt}")
        except Exception as e:
            print(e)


def download_meshes(seg_ids, target_dir, cv_path, overwrite=True,
                    n_threads=1, verbose=False,
                    stitch_mesh_chunks=True, 
                    merge_large_components=False, 
                    map_gs_to_https=True, fmt="hdf5"):
    """ Downloads meshes in target directory (in parallel)

    :param seg_ids: list of uint64s
    :param target_dir: str
    :param cv_path: str
    :param overwrite: bool
    :param n_threads: int
    :param verbose: bool
    :param merge_large_components: bool
    :param fmt: str
        "h5" is highly recommended
    """

    if n_threads > 1:
        n_jobs = n_threads * 3
    else:
        n_jobs = 1

    if len(seg_ids) < n_jobs:
        n_jobs = len(seg_ids)

    seg_id_blocks = np.array_split(seg_ids, n_jobs)

    multi_args = []
    for seg_id_block in seg_id_blocks:
        multi_args.append([seg_id_block, cv_path, target_dir, fmt,
                           overwrite, merge_large_components, stitch_mesh_chunks,
                            map_gs_to_https])

    if n_jobs == 1:
        mu.multiprocess_func(_download_meshes_thread,
                             multi_args, debug=True,
                             verbose=verbose, n_threads=n_threads)
    else:
        mu.multisubprocess_func(_download_meshes_thread,
                                multi_args, n_threads=n_threads,
                                package_name="meshparty", n_retries=40)


class MeshMeta(object):
    def __init__(self, cache_size=400, cv_path=None, disk_cache_path=None,
                 map_gs_to_https=True):
        """ Manager class to keep meshes in memory and seemingless download them

        :param cache_size: int
            adapt this to your available memory
        :param cv_path: str
        :param disk_cache_path: str
            meshes are dumped to this directory => should be equal to target_dir
            in download_meshes
        """
        self._mesh_cache = {}
        self._cache_size = cache_size
        self._cv_path = cv_path
        self._cv = None
        self._map_gs_to_https = map_gs_to_https
        self._disk_cache_path = disk_cache_path

        if self.disk_cache_path is not None:
            if not os.path.exists(self.disk_cache_path):
                os.makedirs(self.disk_cache_path)

    @property
    def cache_size(self):
        return self._cache_size

    @property
    def cv_path(self):
        return self._cv_path

    @property
    def disk_cache_path(self):
        return self._disk_cache_path

    @property
    def cv(self):
        if self._cv is None and self.cv_path is not None:
            self._cv = cloudvolume.CloudVolumeFactory(self.cv_path, parallel=10,
                                        map_gs_to_https=self._map_gs_to_https)

        return self._cv

    def _filename(self, seg_id):
        assert self.disk_cache_path is not None

        return "%s/%d.h5" % (self.disk_cache_path, seg_id)

    def mesh(self, filename=None, seg_id=None, cache_mesh=True,
             merge_large_components=False,
             stitch_mesh_chunks=True,
             overwrite_merge_large_components=False,
             force_download=False):
        """ Loads mesh either from cache, disk or google storage

        :param filename: str
        :param seg_id: uint64
        :param cache_mesh: bool
            if True: mesh is cached in a dictionary. The user is responsible
            for avoiding a memory overflow
        :param merge_large_components: bool
            if True: large (>100 vx) mesh connected components are linked
            and the additional edges strored in .link_edges
            this information is cached as well (default False)
        :param stitch_mesh_chunks: bool
            if True it will stitch the mesh fragments together into a single graph
            (default True)
        :param overwrite_merge_large_components: bool
            if True: recalculate large components
        :return: Mesh
        """
        assert filename is not None or \
            (seg_id is not None and self.cv is not None)

        if filename is not None:
            if filename not in self._mesh_cache:
                mesh_data = read_mesh(filename)
                vertices, faces, normals, link_edges, node_mask = mesh_data
                mesh = Mesh(vertices=vertices, faces=faces, normals=normals,
                                        link_edges=link_edges, node_mask=node_mask)

                if cache_mesh and len(self._mesh_cache) < self.cache_size:
                    self._mesh_cache[filename] = mesh
            else:
                mesh = self._mesh_cache[filename]

            if self.disk_cache_path is not None and \
                    overwrite_merge_large_components:
                write_mesh_h5(filename, mesh.vertices,
                              mesh.faces.flatten(),
                              link_edges=mesh.link_edges)
        else:
            if self.disk_cache_path is not None and force_download is False:
                if os.path.exists(self._filename(seg_id)):
                    mesh = self.mesh(filename=self._filename(seg_id),
                                     cache_mesh=cache_mesh,
                                     merge_large_components=merge_large_components,
                                     overwrite_merge_large_components=overwrite_merge_large_components)
                    return mesh

            if seg_id not in self._mesh_cache or force_download is True:
                cv_mesh = self.cv.mesh.get(seg_id, remove_duplicate_vertices=False)
                faces = np.array(cv_mesh["faces"])
                if (len(faces.shape) == 1):
                    faces = faces.reshape(-1, 3)

                mesh = Mesh(vertices=cv_mesh["vertices"],
                            faces=faces)

                if cache_mesh and len(self._mesh_cache) < self.cache_size:
                    self._mesh_cache[seg_id] = mesh

                if self.disk_cache_path is not None:
                    write_mesh_h5(self._filename(seg_id), mesh.vertices,
                                  mesh.faces,
                                  link_edges=mesh.link_edges,
                                  overwrite=force_download)
            else:
                mesh = self._mesh_cache[seg_id]
    
        if (merge_large_components and (len(mesh.link_edges)==0)) or \
                        overwrite_merge_large_components:
                    mesh.merge_large_components()
        return mesh

class Mesh(trimesh.Trimesh):
    def __init__(self, *args, node_mask=None, unmasked_size=None, apply_mask=False, link_edges=None, **kwargs):
        if 'vertices' in kwargs:
            vertices_all = kwargs.pop('vertices')
        else:
            vertices_all = args[0]

        if 'faces' in kwargs:
            faces_all = kwargs.pop('faces')
        else:
            # If faces are in args, vertices must also have been in args
            faces_all = args[1]

        if unmasked_size is None:
            if node_mask is not None:
                unmasked_size = len(node_mask)
            else:
                unmasked_size = len(vertices_all)
        if unmasked_size < len(vertices_all):
            raise ValueError('Original size cannot be smaller than current size')
        self._unmasked_size = unmasked_size

        if node_mask is None:
            node_mask = np.full(self.unmasked_size, True, dtype=bool)
        elif node_mask.dtype is not np.dtype('bool'):
            node_mask_inds = node_mask.copy()
            node_mask = np.full(self.unmasked_size, False, dtype=bool)
            node_mask[node_mask_inds] = True

        if len(node_mask) != unmasked_size:
            raise ValueError('The node mask must be the same length as the unmasked size')

        self._node_mask = node_mask

        if apply_mask:
            if any(self.node_mask == False):
                nodes_f = vertices_all[self.node_mask]
                faces_f = utils.filter_shapes(np.flatnonzero(node_mask), faces_all)[0]
            else:
                nodes_f, faces_f = vertices_all, faces_all
        else:
            nodes_f, faces_f = vertices_all, faces_all



        new_args = (nodes_f, faces_f)
        if len(args) > 2:
            new_args += args[2:]
        if kwargs.get('process', False):
            print('No silent changing of the mesh is allowed')
        kwargs['process'] = False
        
        super(Mesh, self).__init__(*new_args, **kwargs)
        if apply_mask:
            if link_edges is not None:
                if any(self.node_mask == False):
                    self.link_edges = utils.filter_shapes(np.flatnonzero(node_mask), link_edges)[0]
                else:
                    self.link_edges = link_edges
            else:
                self.link_edges = None
        else:
            self.link_edges = link_edges

        self._index_map = None
        
    @property
    def link_edges(self):
        return self._data['link_edges']

    @link_edges.setter
    def link_edges(self, values):
        if values is None:
            values = np.array([[],[]]).T
        values = np.asanyarray(values, dtype=np.int64)
        # prevents cache from being invalidated
        with self._cache:
            self._data['link_edges']=values
        # now invalidate all items affected
        # not sure this is all of them that are not affected
        # by adding link_edges
        self._cache.clear(exclude=['face_normals',
                                   'vertex_normals',
                                   'faces_sparse',
                                   'bounds',
                                   'extents',
                                   'scale',
                                   'centroid',
                                   'principal_inertia_components',
                                   'principal_inertia_transform',
                                   'symmetry',
                                   'triangles',
                                   'triangles_tree',
                                   'triangles_center',
                                   'triangles_cross',
                                   'edges',
                                   'edges_face',
                                   'edges_unique',
                                   'edges_unique_length'])


    @caching.cache_decorator
    def nxgraph(self):
        return self._create_nxgraph()

    @caching.cache_decorator
    def csgraph(self):
        return self._create_csgraph()

    @caching.cache_decorator
    def pykdtree(self):
        return KDTree(self.vertices)

    @caching.cache_decorator
    def kdtree(self):
        return spatial.cKDTree(self.vertices, balanced_tree=False)

    @property
    def n_vertices(self):
        return len(self.vertices)

    @property
    def n_faces(self):
        return len(self.faces)

    @caching.cache_decorator
    def graph_edges(self):
        return np.vstack([self.edges, self.link_edges])

    def fix_mesh(self, wiggle_vertices=False, verbose=False):
        """ Executes rudimentary fixing function from pymeshfix

        Good for closing holes

        :param wiggle_vertices: bool
            adds robustness for smaller components
        :param verbose: bool
        """
        if self.body_count > 1:
            tin = _meshfix.PyTMesh(verbose)
            # tin.LoadArray(self.vertices, self.faces)
            tin.load_array(self.vertices, self.faces)
            tin.remove_smallest_components()
            # tin.RemoveSmallestComponents()

            # Check if volume is 0 after isolated components have been removed
            # self.vertices, self.faces = tin.ReturnArrays()
            self.vertices, self.faces = tin.return_arrays()

            self.fix_normals()

        if self.volume == 0:
            return

        if wiggle_vertices:
            wiggle = np.random.randn(self.n_vertices * 3).reshape(-1, 3) * 10
            self.vertices += wiggle

        # self.vertices, self.faces = _meshfix.CleanFromVF(self.vertices,
        #                                                  self.faces,
        #                                                  verbose=verbose)
        self.vertices, self.faces = _meshfix.clean_from_arrays(
            self.vertices, self.faces, verbose=verbose)

        self.fix_normals()


    def get_local_views(self, n_points=None,
                        max_dist=np.inf,
                        sample_n_points=None,
                        fisheye=False,
                        pc_align=False,
                        center_node_ids=None,
                        center_coords=None,
                        verbose=False,
                        return_node_ids=False,
                        svd_solver="auto",
                        return_faces=False,
                        adapt_unit_sphere_norm=False,
                        pc_norm=False):
        """ Extracts a local view (points)

        :param n_points: int
            number of points to sample
        :param max_dist: float
            sets an upper limit for distance of any sampled mesh point. Might
            reduce n_points
        :param sample_n_points: int
            has to be >= n_points; if > n_points more points are sampled and a
            subset randomly chosen
        :param fisheye: bool
            addition to sample_n_points; subset is sampled such that a fisheye
            effect is generated
        :param pc_align: bool
            computes PCA and orients mesh along PCs
        :param center_node_ids: list of ints
            mesh vertices at the center of the local views
        :param center_coords: list (n, 3) of floats
            coordinates at the center of the local views
        :param verbose: bool
        :param return_node_ids: bool
            sampled node ids are returned as well, changes the output format
        :param svd_solver: str
            PCA solver
        :param return_faces: bool
            sampled faces are returned as well, changes the output format
        :param adapt_unit_sphere_norm: bool
            NOT FUNCTIONAL
        :param pc_norm: bool
            if True: normalize point cloud to mean 0 and std 1 before PCA
        :return: variable
        """
        if center_node_ids is None and center_coords is None:
            center_node_ids = np.array([np.random.randint(len(self.vertices))])

        if center_coords is None:
            center_node_ids = np.array(center_node_ids, dtype=np.int)
            center_coords = self.vertices[center_node_ids]

        if sample_n_points is None:
            sample_n_points = n_points
        elif sample_n_points > n_points:
            assert not return_faces
        elif sample_n_points == n_points:
            pass
        else:
            raise Exception("Too few sample points specified")

        center_coords = np.array(center_coords)

        if sample_n_points is None:
            sample_n_points = len(self.vertices)
        else:
            sample_n_points = np.min([sample_n_points, len(self.vertices)])

        dists, node_ids = self.kdtree.query(center_coords, sample_n_points,
                                            distance_upper_bound=max_dist)
        if n_points is not None:
            if sample_n_points > n_points:
                if fisheye:
                    probs = 1 / dists

                    new_dists = []
                    new_node_ids = []
                    ids = np.arange(0, sample_n_points, dtype=np.int)
                    for i_sample in range(len(center_coords)):
                        sample_ids = np.random.choice(ids, n_points,
                                                      replace=False,
                                                      p=probs[i_sample])
                        new_dists.append(dists[i_sample, sample_ids])
                        new_node_ids.append(node_ids[i_sample, sample_ids])

                    dists = np.array(new_dists, dtype=np.float32)
                    node_ids = np.array(new_node_ids, dtype=np.int)
                else:
                    ids = np.arange(0, sample_n_points, dtype=np.int)
                    sample_ids = np.random.choice(ids, n_points, replace=False)

                    dists = dists[:, sample_ids]
                    node_ids = node_ids[:, sample_ids]

        if verbose:
            print(np.mean(dists, axis=1), np.max(dists, axis=1),
                  np.min(dists, axis=1))

        if max_dist < np.inf:
            node_ids = list(node_ids)
            local_vertices = []
            for i_ns, ns in enumerate(node_ids):
                ns = ns[dists[i_ns] <= max_dist]
                ns = np.sort(ns)
                node_ids[i_ns] = ns
                local_vertices.append(self.vertices[ns].copy())
        else:
            node_ids = np.sort(node_ids, axis=1)
            local_vertices = self.vertices[node_ids].copy()

        if pc_align:
            for i_lv in range(len(local_vertices)):
                local_vertices[i_lv] = self._calc_pc_align(local_vertices[i_lv],
                                                           svd_solver,
                                                           pc_norm=pc_norm)

        if adapt_unit_sphere_norm:
            local_vertices -= center_coords
            lengths = np.linalg.norm(local_vertices, axis=2)
            local_vertices /= np.max(lengths, axis=1)[:, None, None]

        return_tuple = (local_vertices, center_node_ids)

        if return_node_ids:
            return_tuple += (node_ids, )

        if return_faces:
            return_tuple += (self._filter_faces(node_ids), )

        return return_tuple

    def get_local_view(self, n_points=None, max_dist=np.inf,
                       sample_n_points=None,
                       pc_align=False, center_node_id=None,
                       center_coord=None, method="kdtree", verbose=False,
                       return_node_ids=False, svd_solver="auto",
                       return_faces=False, pc_norm=False):
        """ Single version of get_local_views for backwards compatibility """

        assert method == "kdtree"

        if center_node_id is None and center_coord is None:
            center_node_id = np.random.randint(len(self.vertices))

        if center_coord is None:
            center_coord = self.vertices[center_node_id]

        return self.get_local_views(n_points=n_points,
                                    sample_n_points=sample_n_points,
                                    max_dist=max_dist,
                                    pc_align=pc_align,
                                    center_node_ids=[center_node_id],
                                    center_coords=[center_coord],
                                    verbose=verbose,
                                    return_node_ids=return_node_ids,
                                    svd_solver=svd_solver,
                                    return_faces=return_faces,
                                    pc_norm=pc_norm)


    def _filter_faces(self, node_ids):
        """ node_ids has to be sorted! """
        return utils.filter_shapes(node_ids, self.faces)

    def _filter_graph_edges(self, node_ids):
        """ node_ids has to be sorted! """
        return utils.filter_shapes(node_ids, self.graph_edges)


    def add_link_edges(self, seg_id, dataset_name, close_map_distance=300,
                        server_address="https://www.dynamicannotationframework.com"):
        """ add a set of link edges to this mesh from a pcg endpoint

        :param seg_id: int 
            the seg_id of this mesh
        :param dataset_name: str
            the dataset name this mesh can be found in
        :param close_map_distance: float
            the distance in mesh vertex coordinates to consider a mapping to be 'close'
        :server_address: str
            the server address to find the pcg endpoint (default https://www.dynamicannotationframework.com)
        """
        link_edges = trimesh_repair.get_link_edges(self, seg_id, dataset_name,
                                                   close_map_distance = close_map_distance,
                                                   server_address=server_address)
        self.link_edges = np.vstack([self.link_edges, link_edges])


                        
    def get_local_meshes(self, n_points, max_dist=np.inf, center_node_ids=None,
                         center_coords=None, pc_align=False, pc_norm=False,
                         fix_meshes=False):
        """ Extracts a local mesh

        :param n_points: int
        :param max_dist: float
        :param center_node_ids: list of ints
        :param center_coords: list (n, 3) of floats
        :param pc_align: bool
        :param pc_norm: bool
        :param fix_meshes: bool
        """
        local_view_tuple = self.get_local_views(n_points=n_points,
                                                max_dist=max_dist,
                                                center_node_ids=center_node_ids,
                                                center_coords=center_coords,
                                                return_faces=True,
                                                pc_align=pc_align,
                                                pc_norm=pc_norm)
        vertices, _, faces = local_view_tuple

        meshes = [Mesh(vertices=v, faces=f) for v, f in
                  zip(vertices, faces)]

        if fix_meshes:
            for mesh in meshes:
                if mesh.n_vertices > 0:
                    mesh.fix_mesh(wiggle_vertices=False)

        return meshes

    def get_local_mesh(self, n_points=None, max_dist=np.inf,
                       center_node_id=None,
                       center_coord=None, pc_align=True, pc_norm=False):
        """ Single version of get_local_meshes """

        if center_node_id is not None:
            center_node_id = [center_node_id]

        if center_coord is not None:
            center_coord = [center_coord]

        return self.get_local_meshes(n_points, max_dist=max_dist,
                                     center_node_ids=center_node_id,
                                     center_coords=center_coord,
                                     pc_align=pc_align, pc_norm=pc_norm)[0]

    def _calc_pc_align(self, vertices, svd_solver, pc_norm=False):
        """ Calculates PC alignment """

        vertices = vertices.copy()
        if pc_norm:
            vertices -= vertices.mean(axis=0)
            vertices /= vertices.std(axis=0)

        pca = decomposition.PCA(n_components=3,
                                svd_solver=svd_solver,
                                copy=False)

        return pca.fit_transform(vertices)


    def merge_large_components(self, size_threshold=100, max_dist=1000,
                               dist_step=100):
        """ Finds edges between disconnected components

        :param size_threshold: int
        :param max_dist: float
        """
        time_start = time.time()

        ccs = sparse.csgraph.connected_components(self.csgraph)
        ccs_u, cc_sizes = np.unique(ccs[1], return_counts=True)
        large_cc_ids = ccs_u[cc_sizes > size_threshold]

        print(len(large_cc_ids))

        kdtrees = []
        vertex_ids = []
        for cc_id in large_cc_ids:
            m = ccs[1] == cc_id
            vertex_ids.append(np.where(m)[0])
            v = self.vertices[m]
            kdtrees.append(spatial.cKDTree(v))

        add_edges = []
        for i_tree in range(len(large_cc_ids) - 1):
            for j_tree in range(i_tree + 1, len(large_cc_ids)):
                print("%d - %d      " % (i_tree, j_tree), end="\r")

                if np.any(kdtrees[i_tree].query_ball_tree(kdtrees[j_tree], max_dist)):
                    for this_dist in range(dist_step, max_dist + dist_step, dist_step):

                        pairs = kdtrees[i_tree].query_ball_tree(kdtrees[j_tree],
                                                                this_dist)

                        if np.any(pairs):
                            for i_p, p in enumerate(pairs):
                                if len(p) > 0:
                                    add_edges.extend([[vertex_ids[i_tree][i_p],
                                                       vertex_ids[j_tree][v]]
                                                      for v in p])
                            break

        print(f"Adding {len(add_edges)} new edges.")

        self.link_edges = np.vstack([self.link_edges, add_edges])

        print("TIME MERGING: %.3fs" % (time.time() - time_start))

    def _create_nxgraph(self):
        """ Computes networkx graph """
        return utils.create_nxgraph(self.vertices, self.graph_edges, euclidean_weight=True,
                                    directed=False)

    def _create_csgraph(self):
        """ Computes csgraph """
        return utils.create_csgraph(self.vertices, self.graph_edges, euclidean_weight=True,
                                    directed=False)
    @property
    def node_mask(self):
        '''
        Returns the node mask currently applied to the data
        '''
        return self._node_mask

    @property
    def indices_unmasked(self):
        '''
            Gets the indices of nodes in the filtered mesh in the unmasked array
        '''
        return np.flatnonzero(self.node_mask)

    @property
    def unmasked_size(self):
        '''
        Returns the unmasked number of nodes in the mesh
        '''
        return self._unmasked_size

    def apply_mask(self, new_mask, **kwargs):
        '''
        Makes a new MaskedMesh by adding a new mask to the existing one.
        new_mask is a boolean array, either of the original length or the
        masked length (in which case it is padded with zeros appropriately).
        '''
        # We need to express the mask in the current vertex indices
        if np.size(new_mask) == np.size(self.node_mask):
            joint_mask = self.node_mask & new_mask
            new_mask = self.filter_unmasked_boolean(new_mask)
        elif np.size(new_mask) == self.vertices.shape[0]:
            joint_mask = self.node_mask & self.map_boolean_to_unmasked(new_mask)
        else:
            raise ValueError('Incompatible shape. Must be either original length or current length of vertices.')

        new_mesh = Mesh(self.vertices,
                        self.faces,
                        node_mask=joint_mask,
                        unmasked_size=self.unmasked_size,
                        **kwargs)
        link_edge_unmask = self.map_indices_to_unmasked(self.link_edges)        
        new_mesh._apply_new_mask_in_place(new_mask, link_edge_unmask)
        return new_mesh

    def _apply_new_mask_in_place(self, mask, link_edge_unmask):
        # Use builtin Trimesh tools for masking
        # The new 0 index is the first nonzero element of the mask.
        # Unfortunately, update_vertices maps all masked face values to 0 as well.
        num_zero_expected = np.sum(self.faces==np.flatnonzero(mask)[0], axis=1)
        self.update_vertices(mask)

        num_zero_new = np.sum(self.faces==0, axis=1)
        faces_to_keep = num_zero_new == num_zero_expected
        self.update_faces(faces_to_keep)
        self.link_edges = self.filter_unmasked_indices(link_edge_unmask)


    def map_indices_to_unmasked(self, unmapped_indices):
        '''
        For a set of masked indices, returns the corresponding unmasked indices
        '''
        return self.indices_unmasked[unmapped_indices]

    def map_boolean_to_unmasked(self, unmapped_boolean):
        '''
        For a boolean index in the masked indices, returns the corresponding unmasked boolean index
        '''
        full_boolean = np.full(self.unmasked_size, False)
        full_boolean[self.node_mask] = unmapped_boolean
        return full_boolean

    def filter_unmasked_boolean(self, unmasked_boolean):
        '''
        For an unmasked boolean slice, returns a boolean slice filtered to the masked mesh
        '''
        return unmasked_boolean[self.node_mask]

    def filter_unmasked_indices(self, unmasked_shape, mask=None):
        if mask is None:
            mask = self.node_mask
        new_index = np.zeros(mask.shape)-1
        new_index[mask] = np.arange(np.sum(mask))
        new_shape = new_index[unmasked_shape.ravel()].reshape(unmasked_shape.shape).astype(int)
        
        if len(new_shape.shape) > 1:
            keep_rows = np.all(new_shape>=0, axis=1)
        else:
            keep_rows = new_shape>=0

        return new_shape[keep_rows]

    def write_to_file(self, filename):
        """ Exports the mesh to any format supported by trimesh

        :param filename: str
        """
        if os.path.splitext(filename)[1]=='.h5':
            write_mesh_h5(filename,
                          self.vertices,
                          self.faces,
                          normals=self.face_normals,
                          link_edges=self.link_edges,
                          node_mask=self.node_mask,
                          overwrite=True)
        else:
            exchange.export.export_mesh(self, filename)
    @property
    def index_map(self):
        '''
        A dict mapping global indices into the masked mesh indices.
        '''
        if self._index_map is None:
            self._index_map = defaultdict(lambda: np.nan)
            for ii, index in enumerate(self.indices_unmasked):
                self._index_map[index] = ii
        return self._index_map

class MaskedMesh(Mesh):
    def __init__(self, *args,  **kwargs):
        warnings.warn(
            "use of MaskedMesh deprecated, Mesh now contains all MaskedMesh functionality",
            DeprecationWarning)
        super(MaskedMesh, self).__init__(*args, **kwargs)
