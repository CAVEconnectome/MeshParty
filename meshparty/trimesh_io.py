import numpy as np
import h5py
from scipy import spatial
from sklearn import decomposition
import plyfile
import os
import networkx as nx
import requests

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

    print(node_id, url)
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

        print("frags", frags)

        try:
            if fmt == "hdf5":
                mesh = cv.mesh.get(frags)
                write_mesh_h5(f"{seg_id}.h5", mesh["vertices"], mesh["faces"],
                              overwrite=overwrite)
            elif fmt == "obj":
                cv.mesh.save(frags)
            else:
                raise Exception(f"unknown fmt: {fmt}")
        except Exception as e:
            print(e)

def download_meshes(seg_ids, target_dir, cv_path, overwrite=True,
                    mesh_endpoint=None, n_threads=1, verbose=False, fmt="obj"):
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
    def __init__(self):
        self.filename_dict = {}

    def mesh(self, filename):
        if not filename in self.filename_dict:
            # print("reload -- elements in cache: %d" % len(self.filename_dict))
            # print(self.filename_dict)
            # try:
            if len(self.filename_dict) > 400:
                self.filename_dict = {}

            if filename.endswith(".obj"):
                vertices, faces, normals = read_mesh_obj(filename)
            elif filename.endswith(".h5"):
                vertices, faces, normals = read_mesh_h5(filename)
            else:
                raise Exception("Unknown filetype")

            mesh = Mesh(vertices=vertices, faces=faces, normals=normals)

            self.filename_dict[filename] = mesh
            # except:
            #     self.filename_dict[filename] = None

        return self.filename_dict[filename]


class Mesh(trimesh.Trimesh):
    def __init__(self, *args, **kwargs):
        super(Mesh, self).__init__(*args, **kwargs)

    @caching.cache_decorator
    def graph(self):
        graph = self.create_nx_graph()
        return graph

    def fix_mesh(self):
        vclean, fclean = _meshfix.CleanFromVF(self.vertices,
                                              self.faces)

        self.vertices = vclean
        self.faces = fclean

    def write_h5(self, overwrite=False):
        """Writes data to an hdf5 file"""
        normals = None if len(self.normals) == 0 else self.normals

        write_mesh_h5(self.filename, self.vertices, self.faces,
                      normals, overwrite=overwrite)

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


    def get_local_view(self, n_points, pc_align=False, center_node_id=None,
                       center_coord=None, method="kdtree", verbose=False,
                       return_node_ids=False, svd_solver="auto",
                       return_faces=False, pc_norm=False):
        if center_node_id is None and center_coord is None:
            center_node_id = np.random.randint(len(self.vertices))

        if center_coord is None:
            center_coord = self.vertices[center_node_id]

        n_samples = np.min([n_points, len(self.vertices)])

        if method == "kdtree":
            dists, node_ids = self.kdtree.query(center_coord, n_samples,
                                                n_jobs=-1)
            if verbose:
                print(np.mean(dists), np.max(dists), np.min(dists))
        elif method == "graph":
           dist_dict = nx.single_source_dijkstra_path_length(self.graph,
                                                             center_node_id,
                                                             weight="weight")
           sorting = np.argsort(np.array(list(dist_dict.values())))
           node_ids = np.array(list(dist_dict.keys()))[sorting[:n_points]]
        else:
            raise Exception("unknow method")

        local_vertices = self.vertices[node_ids].copy()

        if pc_align:
            local_vertices = self.calc_pc_align(local_vertices, svd_solver,
                                                pc_norm=pc_norm)

        return_tuple = (local_vertices, center_node_id)

        if return_node_ids:
            return_tuple += (node_ids, )

        if return_faces:
            return_tuple += (self._filter_faces(node_ids), )

        return return_tuple

    def _filter_faces(self, node_ids):
        def _remap(entry):
            return mapper_dict[entry] if entry in mapper_dict else entry

        filtered_faces = self.faces.copy()

        filtered_faces = filtered_faces[np.in1d(filtered_faces[:, 0], node_ids)]
        filtered_faces = filtered_faces[np.in1d(filtered_faces[:, 1], node_ids)]
        filtered_faces = filtered_faces[np.in1d(filtered_faces[:, 2], node_ids)]

        mapper_dict = dict(zip(node_ids,
                               np.arange(len(node_ids), dtype=np.int)))
        _remap = np.vectorize(_remap)
        filtered_faces = _remap(filtered_faces)

        return filtered_faces

    def get_local_mesh(self, n_points, center_node_id=None, center_coord=None,
                       method="kdtree", pc_align=True, pc_norm=False):
        vertices, _, faces = self.get_local_view(n_points=n_points,
                                                 center_node_id=center_node_id,
                                                 center_coord=center_coord,
                                                 method=method,
                                                 return_faces=True,
                                                 pc_align=pc_align,
                                                 pc_norm=pc_norm)

        return Mesh(vertices=vertices, faces=faces)

    def calc_pc_align(self, vertices, svd_solver, pc_norm=False):
        if pc_norm:
            vertices -= vertices.mean(axis=0)
            vertices /= vertices.std(axis=0)
        pca = decomposition.PCA(n_components=3, svd_solver=svd_solver,
                                copy=False)
        return pca.fit_transform(vertices)

    def create_nx_graph(self):
        weights = np.linalg.norm(self.vertices[self.edges[:, 0]] -
                                 self.vertices[self.edges[:, 1]], axis=1)

        print(weights.shape)

        weighted_graph = nx.Graph()
        weighted_graph.add_edges_from(self.edges)

        for i_edge, edge in enumerate(self.edges):
            weighted_graph[edge[0]][edge[1]]['weight'] = weights[i_edge]

        return weighted_graph


