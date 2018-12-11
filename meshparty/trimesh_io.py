import numpy as np
import h5py
from scipy import spatial, sparse
from sklearn import decomposition
import plyfile
import os
import networkx as nx
import requests
import time

import cloudvolume
from multiwrapper import multiprocessing_utils as mu

import trimesh
from trimesh import caching, io

from pymeshfix import _meshfix


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

    return vertices, faces, normals


def write_mesh_h5(filename, vertices, faces,
                  normals=None, overwrite=False):
    """Writes a mesh's vertices, faces (and normals) to an hdf5 file"""

    if os.path.isfile(filename):
        if overwrite:
            os.remove(filename)
        else:
            raise Exception(f"File {filename} already exists")

    with h5py.File(filename, "w") as f:
        f.create_dataset("vertices", data=vertices, compression="gzip")
        f.create_dataset("faces", data=faces, compression="gzip")

        if normals is not None:
            f.create_dataset("normals", data=normals, compression="gzip")


def read_mesh(filename):
    """Reads a mesh's vertices, faces and normals from obj or h5 file"""

    if filename.endswith(".obj"):
        vertices, faces, normals = read_mesh_obj(filename)
    elif filename.endswith(".h5"):
        vertices, faces, normals = read_mesh_h5(filename)
    else:
        raise Exception("Unknown filetype")

    return vertices, faces, normals


def read_mesh_obj(filename):
    """Reads a mesh's vertices, faces and normals from an obj file"""
    vertices = []
    faces = []
    normals = []

    for line in open(filename, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
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
    url = "%s/1.0/%d/validfragments" % (endpoint, node_id)
    r = requests.get(url)

    assert r.status_code == 200

    frag_ids = np.frombuffer(r.content, dtype=np.uint64)

    return list(frag_ids)


def _download_meshes_thread(args):
    """ Downloads meshes into target directory

    :param args: list
    """
    seg_ids, cv_path, target_dir, fmt, overwrite, mesh_endpoint = args

    cv = cloudvolume.CloudVolume(cv_path)
    os.chdir(target_dir)

    for seg_id in seg_ids:
        if not overwrite and os.path.exists(f"{seg_id}.h5"):
            continue

        frags = [np.uint64(seg_id)]

        if mesh_endpoint is not None:
            frags = get_frag_ids_from_endpoint(seg_id, mesh_endpoint)

        try:
            mesh = cv.mesh.get(frags)
            if fmt == "hdf5":
                write_mesh_h5(f"{seg_id}.h5", mesh["vertices"], mesh["faces"],
                              overwrite=overwrite)
            else:
                mesh.write_to_file(f"{seg_id}.{fmt}")
        except Exception as e:
            print(e)


def download_meshes(seg_ids, target_dir, cv_path, overwrite=True,
                    mesh_endpoint=None, n_threads=1, verbose=False,
                    straggler_detection=True, fmt="hdf5"):
    """ Downloads meshes in target directory (parallel)

    :param seg_ids: list of ints
    :param target_dir: str
    :param cv_path: str
    :param n_threads: int
    :param fmt: str, desired file format ("obj" or "hdf5")
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
                           overwrite, mesh_endpoint])

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
                 mesh_endpoint=None):
        self._mesh_cache = {}
        self._cache_size = cache_size
        self._cv_path = cv_path
        self._cv = None
        self._disk_cache_path = disk_cache_path
        self._mesh_endpoint = mesh_endpoint

        if not self.disk_cache_path is None:
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
    def mesh_endpoint(self):
        return self._mesh_endpoint

    @property
    def cv(self):
        if self._cv is None and self.cv_path is not None:
            self._cv = cloudvolume.CloudVolume(self.cv_path, parallel=10)

        return self._cv

    def filename(self, seg_id):
        assert self.disk_cache_path is not None

        return "%s/%d.h5" % (self.disk_cache_path, seg_id)

    def mesh(self, filename=None, seg_id=None, cache_mesh=True, fix_mesh=False):
        assert filename is not None or \
               (seg_id is not None and self.cv is not None)

        if filename is not None:
            if not filename in self._mesh_cache:
                vertices, faces, normals = read_mesh(filename)

                mesh = Mesh(vertices=vertices, faces=faces, normals=normals)

                if fix_mesh:
                    mesh.fix_mesh()

                if cache_mesh and len(self._mesh_cache) < self.cache_size:
                    self._mesh_cache[filename] = mesh
            else:
                mesh = self._mesh_cache[filename]
        else:
            if self.disk_cache_path is not None:
                if os.path.exists(self.filename(seg_id)):
                    return self.mesh(filename=self.filename(seg_id),
                                     cache_mesh=cache_mesh)

            if not seg_id in self._mesh_cache:
                if self.mesh_endpoint is not None:
                    frags = get_frag_ids_from_endpoint(seg_id,
                                                       self.mesh_endpoint)
                else:
                    frags = [seg_id]

                cv_mesh = self.cv.mesh.get(frags)

                mesh = Mesh(vertices=cv_mesh["vertices"],
                            faces=np.array(cv_mesh["faces"]).reshape(-1, 3))

                if fix_mesh:
                    mesh.fix_mesh()

                if cache_mesh and len(self._mesh_cache) < self.cache_size:
                    self._mesh_cache[seg_id] = mesh

                if self.disk_cache_path is not None:
                    write_mesh_h5(self.filename(seg_id), mesh.vertices,
                                  mesh.faces.flatten())
            else:
                mesh = self._mesh_cache[seg_id]

        return mesh


class Mesh(trimesh.Trimesh):
    def __init__(self, *args, **kwargs):
        super(Mesh, self).__init__(*args, **kwargs)

    @caching.cache_decorator
    def graph(self):
        graph = self.create_nx_graph()
        return graph

    @caching.cache_decorator
    def csgraph(self):
        csgraph = self.create_csgraph()
        return csgraph

    @property
    def n_vertices(self):
        return len(self.vertices)

    @property
    def n_faces(self):
        return len(self.faces)

    def fix_mesh(self, wiggle_vertices=False, verbose=False):

        tin = _meshfix.PyTMesh(verbose)
        tin.LoadArray(self.vertices, self.faces)
        tin.RemoveSmallestComponents()

        # Check if volume is 0 after isolated components have been removed
        self.vertices, self.faces = tin.ReturnArrays()

        self.fix_normals()

        if self.volume == 0:
            return

        if wiggle_vertices:
            self.vertices += np.random.rand(self.n_vertices * 3).reshape(-1, 3) * 10

        self.vertices, self.faces = _meshfix.CleanFromVF(self.vertices,
                                                         self.faces,
                                                         verbose=False)

        self.fix_normals()

    def write_vertices_ply(self, out_fname, coords=None):
        """Writing vertex coordinates as a .ply file using plyfile"""

        if coords is None:
            coords = self.vertices

        tweaked_array = np.array(
            list(zip(coords[:, 0], coords[:, 1], coords[:, 2])),
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

        vertex_element = plyfile.PlyElement.describe(tweaked_array, "vertex")

        if not os.path.exists(os.path.dirname(out_fname)):
            os.makedirs(os.path.dirname(out_fname))

        plyfile.PlyData([vertex_element]).write(out_fname)

    def write_to_file(self, filename):
        io.export.export_mesh(self, filename)

    def get_local_views(self, n_points, pc_align=False, center_node_ids=None,
                        center_coords=None, verbose=False,
                        return_node_ids=False, svd_solver="auto",
                        return_faces=False, pc_norm=False):

        if center_node_ids is None and center_coords is None:
            center_node_ids = np.array([np.random.randint(len(self.vertices))])

        if center_coords is None:
            center_node_ids = np.array(center_node_ids, dtype=np.int)
            center_coords = self.vertices[center_node_ids]

        center_coords = np.array(center_coords)

        n_samples = np.min([n_points, len(self.vertices)])

        dists, node_ids = self.kdtree.query(center_coords, n_samples,
                                            n_jobs=-1)
        if verbose:
            print(np.mean(dists, axis=1), np.max(dists, axis=1),
                  np.min(dists, axis=1))

        node_ids = np.sort(node_ids, axis=1)

        local_vertices = self.vertices[node_ids]

        if pc_align:
            for i_lv in range(len(local_vertices)):
                local_vertices[i_lv] = self.calc_pc_align(local_vertices[i_lv],
                                                          svd_solver,
                                                          pc_norm=pc_norm)

        return_tuple = (local_vertices, center_node_ids)

        if return_node_ids:
            return_tuple += (node_ids, )

        if return_faces:
            return_tuple += (self._filter_faces(node_ids), )

        return return_tuple

    def get_local_view(self, n_points, pc_align=False, center_node_id=None,
                       center_coord=None, method="kdtree", verbose=False,
                       return_node_ids=False, svd_solver="auto",
                       return_faces=False, pc_norm=False):

        assert method == "kdtree"

        if center_node_id is None and center_coord is None:
            center_node_id = np.random.randint(len(self.vertices))

        if center_coord is None:
            center_coord = self.vertices[center_node_id]

        return self.get_local_views(n_points=n_points, pc_align=pc_align,
                                    center_node_ids=[center_node_id],
                                    center_coords=[center_coord],
                                    verbose=verbose,
                                    return_node_ids=return_node_ids,
                                    svd_solver=svd_solver,
                                    return_faces=return_faces,
                                    pc_norm=pc_norm)

    def _filter_faces(self, node_ids):
        """ node_ids has to be sorted! """
        if len(node_ids.shape) == 1:
            node_ids = node_ids[None]

        all_node_ids = node_ids.flatten()
        pre_filtered_faces = self.faces[np.in1d(self.faces[:, 0], all_node_ids)]
        pre_filtered_faces = pre_filtered_faces[np.in1d(pre_filtered_faces[:, 1], all_node_ids)]
        pre_filtered_faces = pre_filtered_faces[np.in1d(pre_filtered_faces[:, 2], all_node_ids)]

        filtered_faces = []

        for ns in node_ids:
            f = pre_filtered_faces[np.in1d(pre_filtered_faces[:, 0], ns)]
            f = f[np.in1d(f[:, 1], ns)]
            f = f[np.in1d(f[:, 2], ns)]

            f = np.unique(np.concatenate([f.flatten(), ns]),
                          return_inverse=True)[1][:-len(ns)].reshape(-1, 3)

            filtered_faces.append(f)

        return filtered_faces

    def get_local_meshes(self, n_points, center_node_ids=None,
                         center_coords=None, pc_align=True, pc_norm=False,
                         fix_meshes=False):
        local_view_tuple = self.get_local_views(n_points=n_points,
                                                center_node_ids=center_node_ids,
                                                center_coords=center_coords,
                                                return_faces=True,
                                                pc_align=pc_align,
                                                pc_norm=pc_norm)
        vertices, _, faces = local_view_tuple

        meshes = [Mesh(vertices=v, faces=f) for v, f in zip(vertices, faces)]

        if fix_meshes:
            for mesh in meshes:
                mesh.fix_mesh(wiggle_vertices=True)

        return meshes

    def get_local_mesh(self, n_points, center_node_id=None, center_coord=None,
                       pc_align=True, pc_norm=False):
        if center_node_id is not None:
            center_node_id = [center_node_id]

        if center_coord is not None:
            center_coord = [center_coord]

        return self.get_local_meshes(n_points,
                                     center_node_ids=center_node_id,
                                     center_coords=center_coord,
                                     pc_align=pc_align, pc_norm=pc_norm)[0]

    def calc_pc_align(self, vertices, svd_solver, pc_norm=False):
        vertices = vertices.copy()
        if pc_norm:
            vertices -= vertices.mean(axis=0)
            vertices /= vertices.std(axis=0)

        pca = decomposition.PCA(n_components=3,
                                svd_solver=svd_solver,
                                copy=False)

        return pca.fit_transform(vertices)

    def create_nx_graph(self):
        weights = np.linalg.norm(self.vertices[self.edges[:, 0]] -
                                 self.vertices[self.edges[:, 1]], axis=1)

        weighted_graph = nx.Graph()
        weighted_graph.add_edges_from(self.edges)

        for i_edge, edge in enumerate(self.edges):
            weighted_graph[edge[0]][edge[1]]['weight'] = weights[i_edge]
            weighted_graph[edge[1]][edge[0]]['weight'] = weights[i_edge]

        return weighted_graph

    def create_csgraph(self):
        weights = np.linalg.norm(self.vertices[self.edges[:, 0]] -
                                 self.vertices[self.edges[:, 1]], axis=1)

        edges = np.concatenate([self.edges.T, self.edges.T[[1, 0]]], axis=1)
        weights = np.concatenate([weights, weights]).astype(dtype=np.float32)

        csgraph = sparse.csr_matrix((weights, edges),
                                    shape=[len(self.vertices), ] * 2,
                                    dtype=np.float32)

        return csgraph


