import numpy as np
import h5py
from scipy import spatial, sparse
from sklearn import decomposition
import os
import networkx as nx
import requests
import time

import cloudvolume
from multiwrapper import multiprocessing_utils as mu

import trimesh
try:
    from trimesh import exchange
except ImportError:
    from trimesh import io as exchange

from pymeshfix import _meshfix
from tqdm import trange

from meshparty import utils


def read_mesh_h5(filename):
    """Reads a mesh's vertices, faces and normals from an hdf5 file"""
    assert os.path.isfile(filename)

    with h5py.File(filename, "r") as f:
        vertices = f["vertices"].value
        faces = f["faces"].value

        if len(faces.shape) == 1:
            faces = faces.reshape(-1, 3)

        if "normals" in f.keys():
            normals = f["normals"].value
        else:
            normals = []

        if "mesh_edges" in f.keys():
            mesh_edges = f["mesh_edges"].value
        else:
            mesh_edges = None

    return vertices, faces, normals, mesh_edges


def write_mesh_h5(filename, vertices, faces,
                  normals=None, mesh_edges=None, overwrite=False):
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

        if mesh_edges is not None:
            f.create_dataset("mesh_edges", data=mesh_edges, compression="gzip")


def read_mesh(filename):
    """Reads a mesh's vertices, faces and normals from obj or h5 file"""

    if filename.endswith(".obj"):
        vertices, faces, normals = read_mesh_obj(filename)
        mesh_edges = None
    elif filename.endswith(".h5"):
        vertices, faces, normals, mesh_edges = read_mesh_h5(filename)
    else:
        raise Exception("Unknown filetype")

    return vertices, faces, normals, mesh_edges


def read_mesh_obj(filename):
    """Reads a mesh's vertices, faces and normals from an obj file"""
    vertices = []
    faces = []
    normals = []

    for line in open(filename, "r"):
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'v':
            v = values[1:4]
            vertices.append(v)
        elif values[0] == 'vn':
            v = map(float, values[1:4])
            normals.append(v)
        elif values[0] == 'f':
            face = []
            texcoords = []
            norms = []
            for v in values[1:]:
                w = v.split('/')
                face.append(int(w[0]))
                if len(w) >= 2 and len(w[1]) > 0:
                    texcoords.append(int(w[1]))
                else:
                    texcoords.append(0)
                if len(w) >= 3 and len(w[2]) > 0:
                    norms.append(int(w[2]))
                else:
                    norms.append(0)
            faces.append(face)

    vertices = np.array(vertices, dtype=np.float)
    faces = np.array(faces, dtype=np.int) - 1
    normals = np.array(normals, dtype=np.float)

    return vertices, faces, normals


def get_frag_ids_from_endpoint(node_id, endpoint):
    """ Reads the mesh fragments from the chunkedgraph endpoint """
    url = "%s/1.0/%d/validfragments" % (endpoint, node_id)
    r = requests.get(url)

    assert r.status_code == 200

    frag_ids = np.frombuffer(r.content, dtype=np.uint64)

    return list(frag_ids)


def _download_meshes_thread(args):
    """ Helper to Download meshes into target directory """
    seg_ids, cv_path, target_dir, fmt, overwrite, \
        merge_large_components, remove_duplicate_vertices = args

    cv = cloudvolume.CloudVolumeFactory(cv_path)

    for seg_id in seg_ids:
        print('downloading {}'.format(seg_id))
        target_file = os.path.join(target_dir, f"{seg_id}.h5")
        if not overwrite and os.path.exists(target_file):
            print('file exists {}'.format(target_file))
            continue
        print('file does not exist {}'.format(target_file))
       
        try:
            cv_mesh = cv.mesh.get([seg_id],
                                  remove_duplicate_vertices=remove_duplicate_vertices)

            faces = np.array(cv_mesh["faces"])
            if len(faces.shape) == 1:
                faces = faces.reshape(-1, 3)

            mesh = Mesh(vertices=cv_mesh["vertices"],
                        faces=faces,
                        process=remove_duplicate_vertices)

            if merge_large_components:
                mesh.merge_large_components()

            if fmt == "hdf5":
                write_mesh_h5(f"{target_dir}/{seg_id}.h5", mesh.vertices,
                              mesh.faces.flatten(),
                              mesh_edges=mesh.mesh_edges,
                              overwrite=overwrite)
            else:
                mesh.write_to_file(f"{target_dir}/{seg_id}.{fmt}")
        except Exception as e:
            print(e)


def download_meshes(seg_ids, target_dir, cv_path, overwrite=True,
                    n_threads=1, verbose=False,
                    merge_large_components=True, remove_duplicate_vertices=True,
                    fmt="hdf5"):
    """ Downloads meshes in target directory (in parallel)

    :param seg_ids: list of uint64s
    :param target_dir: str
    :param cv_path: str
    :param overwrite: bool
    :param n_threads: int
    :param verbose: bool
    :param merge_large_components: bool
    :param remove_duplicate_vertices: bool
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
                           overwrite, merge_large_components,
                           remove_duplicate_vertices])

    if n_jobs == 1:
        mu.multiprocess_func(_download_meshes_thread,
                             multi_args, debug=True,
                             verbose=verbose, n_threads=n_threads)
    else:
        mu.multisubprocess_func(_download_meshes_thread,
                                multi_args, n_threads=n_threads,
                                package_name="meshparty", n_retries=40)


class MeshMeta(object):
    def __init__(self, cache_size=400, cv_path=None, disk_cache_path=None):
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
            self._cv = cloudvolume.CloudVolumeFactory(self.cv_path, parallel=10)

        return self._cv

    def _filename(self, seg_id):
        assert self.disk_cache_path is not None

        return "%s/%d.h5" % (self.disk_cache_path, seg_id)

    def mesh(self, filename=None, seg_id=None, cache_mesh=True,
             merge_large_components=True, remove_duplicate_vertices=True,
             overwrite_merge_large_components=False):
        """ Loads mesh either from cache, disk or google storage

        :param filename: str
        :param seg_id: uint64
        :param cache_mesh: bool
            if True: mesh is cached in a dictionary. The user is responsible
            for avoiding a memory overflow
        :param merge_large_components: bool
            if True: large (>100 vx) mesh connected components are linked
            and the additional edges strored in .mesh_edges
            this information is cached as well
        :param remove_duplicate_vertices: bool
            if True will merge vertices with the same coordinates and also
            remove Nan and Inf values through trimesh process=True functionality
        :param overwrite_merge_large_components: bool
            if True: recalculate large components
        :return: Mesh
        """
        assert filename is not None or \
            (seg_id is not None and self.cv is not None)

        if filename is not None:
            if filename not in self._mesh_cache:
                vertices, faces, normals, mesh_edges = read_mesh(filename)

                mesh = Mesh(vertices=vertices, faces=faces, normals=normals,
                            mesh_edges=mesh_edges,
                            process=remove_duplicate_vertices)

                if (merge_large_components and mesh.mesh_edges is None) or \
                        overwrite_merge_large_components:
                    mesh.merge_large_components()

                if cache_mesh and len(self._mesh_cache) < self.cache_size:
                    self._mesh_cache[filename] = mesh
            else:
                mesh = self._mesh_cache[filename]

            if self.disk_cache_path is not None and \
                    overwrite_merge_large_components:
                write_mesh_h5(filename, mesh.vertices,
                              mesh.faces.flatten(),
                              mesh_edges=mesh.mesh_edges)
        else:
            if self.disk_cache_path is not None:
                if os.path.exists(self._filename(seg_id)):
                    return self.mesh(filename=self._filename(seg_id),
                                     cache_mesh=cache_mesh,
                                     merge_large_components=merge_large_components,
                                     overwrite_merge_large_components=overwrite_merge_large_components,
                                     remove_duplicate_vertices=remove_duplicate_vertices)

            if seg_id not in self._mesh_cache:
                cv_mesh = self.cv.mesh.get([seg_id],
                                           remove_duplicate_vertices= remove_duplicate_vertices)
                faces = np.array(cv_mesh["faces"])
                if (len(faces.shape) == 1):
                    faces = faces.reshape(-1, 3)
                mesh = Mesh(vertices=cv_mesh["vertices"],
                            faces=faces,
                            process=remove_duplicate_vertices)

                if (merge_large_components and mesh.mesh_edges is None) or \
                        overwrite_merge_large_components:
                    mesh.merge_large_components()

                if cache_mesh and len(self._mesh_cache) < self.cache_size:
                    self._mesh_cache[seg_id] = mesh

                if self.disk_cache_path is not None:
                    write_mesh_h5(self._filename(seg_id), mesh.vertices,
                                  mesh.faces,
                                  mesh_edges=mesh.mesh_edges)
            else:
                mesh = self._mesh_cache[seg_id]

        return mesh


class Mesh(trimesh.Trimesh):
    def __init__(self, *args, mesh_edges=None, **kwargs):
        super(Mesh, self).__init__(*args, **kwargs)

        self._mesh_edges = mesh_edges
        self._csgraph = None
        self._nxgraph = None

    @property
    def nxgraph(self):
        if self._nxgraph is None:
            self._nxgraph = self._create_nxgraph()
        return self._nxgraph

    @property
    def csgraph(self):
        if self._csgraph is None:
            self._csgraph = self._create_csgraph()
        return self._csgraph

    @property
    def n_vertices(self):
        return len(self.vertices)

    @property
    def n_faces(self):
        return len(self.faces)

    @property
    def mesh_edges(self):
        return self._mesh_edges

    def fix_mesh(self, wiggle_vertices=False, verbose=False):
        """ Executes rudimentary fixing function from pymeshfix

        Good for closing holes

        :param wiggle_vertices: bool
            adds robustness for smaller components
        :param verbose: bool
        """
        if self.body_count > 1:
            tin = _meshfix.PyTMesh(verbose)
            tin.LoadArray(self.vertices, self.faces)
            tin.RemoveSmallestComponents()

            # Check if volume is 0 after isolated components have been removed
            self.vertices, self.faces = tin.ReturnArrays()

            self.fix_normals()

        if self.volume == 0:
            return

        if wiggle_vertices:
            wiggle = np.random.randn(self.n_vertices * 3).reshape(-1, 3) * 10
            self.vertices += wiggle

        self.vertices, self.faces = _meshfix.CleanFromVF(self.vertices,
                                                         self.faces,
                                                         verbose=verbose)

        self.fix_normals()

    def write_to_file(self, filename):
        """ Exports the mesh to any format supported by trimesh

        :param filename: str
        """
        exchange.export.export_mesh(self, filename)

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
                                            distance_upper_bound=max_dist,
                                            n_jobs=-1)
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

    def _filter_shapes(self, node_ids, shapes):
        """ node_ids has to be sorted! """
        if not isinstance(node_ids[0], list) and \
                not isinstance(node_ids[0], np.ndarray):
            node_ids = [node_ids]
        ndim = shapes.shape[1]
        if isinstance(node_ids, np.ndarray):
            all_node_ids = node_ids.flatten()
        else:
            all_node_ids = np.concatenate(node_ids)

        filter_ = np.in1d(shapes[:, 0], all_node_ids)
        pre_filtered_shapes = shapes[filter_].copy()
        for k in range(1, ndim):
            filter_ = np.in1d(pre_filtered_shapes[:, k], all_node_ids)
            pre_filtered_shapes = pre_filtered_shapes[filter_]

        filtered_shapes = []

        for ns in node_ids:
            f = pre_filtered_shapes[np.in1d(pre_filtered_shapes[:, 0], ns)]
            for k in range(1, ndim):
                f = f[np.in1d(f[:, k], ns)]

            f = np.unique(np.concatenate([f.flatten(), ns]),
                          return_inverse=True)[1][:-len(ns)].reshape(-1, ndim)

            filtered_shapes.append(f)

        return filtered_shapes

    def _filter_faces(self, node_ids):
        """ node_ids has to be sorted! """
        return self._filter_shapes(node_ids, self.faces)

    def _filter_mesh_edges(self, node_ids):
        """ node_ids has to be sorted! """
        return self._filter_shapes(node_ids, self.mesh_edges)

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

        self._mesh_edges = None
        self._csgraph = None

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

        if len(add_edges) > 0:
            self._mesh_edges = np.concatenate([self.edges, add_edges])
            self._csgraph = None
            self._nxgraph = None
        else:
            self._mesh_edges = self.edges.copy()

        print("TIME MERGING: %.3fs" % (time.time() - time_start))

    def _create_nxgraph(self):
        """ Computes networkx graph """
        if self.mesh_edges is not None:
            edges = self.mesh_edges
        else:
            edges = self.edges

        return utils.create_nxgraph(self.vertices, edges, euclidean_weight=True,
                                    directed=False)

    def _create_csgraph(self):
        """ Computes csgraph """
        if self.mesh_edges is not None:
            edges = self.mesh_edges
        else:
            edges = self.edges

        return utils.create_csgraph(self.vertices, edges, euclidean_weight=True,
                                    directed=False)
