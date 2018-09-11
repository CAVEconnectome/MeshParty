import numpy as np
import h5py
from scipy import spatial
from sklearn import decomposition
import plyfile
import os
import networkx as nx

import cloudvolume
from multiwrapper import multiprocessing_utils as mu

def read_mesh_h5():
    pass

def write_mesh_h5():
    pass

def read_obj(path):
    return Mesh(path)

def _download_meshes_thread(args):
    """ Downloads meshes into target directory

    :param args: list
    """
    seg_ids, cv_path, target_dir = args

    cv = cloudvolume.CloudVolume(cv_path)
    os.chdir(target_dir)

    for seg_id in seg_ids:
        cv.mesh.save(seg_id)


def download_meshes(seg_ids, target_dir, cv_path, n_threads=1):
    """ Downloads meshes in target directory (parallel)

    :param seg_ids: list of ints
    :param target_dir: str
    :param cv_path: str
    :param n_threads: int
    """

    n_jobs = n_threads * 3
    if len(seg_ids) < n_jobs:
        n_jobs = len(seg_ids)

    seg_id_blocks = np.array_split(seg_ids, n_jobs)

    multi_args = []
    for seg_id_block in seg_id_blocks:
        multi_args.append([seg_id_block, cv_path, target_dir])

    if n_jobs == 1:
        mu.multiprocess_func(_download_meshes_thread,
                             multi_args, debug=True,
                             verbose=True, n_threads=1)
    else:
        mu.multisubprocess_func(_download_meshes_thread,
                                multi_args, n_threads=n_threads)


def refine_mesh():
    pass


class MeshMeta(object):
    def __init__(self):
        self.filename_dict = {}

    def mesh(self, filename):
        if not filename in self.filename_dict:
            try:
                self.filename_dict[filename] = Mesh(filename)
            except:
                self.filename_dict[filename] = None

        return self.filename_dict[filename]

class Mesh(object):
    def __init__(self, filename):
        self._vertices = []
        self._normals = []
        self._faces = []
        self._filename = filename

        self._kdtree = None
        self._graph = None
        self._edges = None

        if not os.path.exists(filename):
            raise Exception("File does not exist")

        if filename.endswith(".obj"):
            self.load_obj()
        elif filename.endswith(".h5"):
            self.load_h5()
        else:
            raise Exception("Unknown filetype")

    @property
    def filename(self):
        return self._filename

    @property
    def vertices(self):
        return self._vertices

    @property
    def faces(self):
        return self._faces

    @property
    def normals(self):
        return self._normals

    @property
    def edges(self):
        if self._edges is None:
            self._edges = np.concatenate([self.faces[:, :2],
                                          self.faces[:, 1:3]], axis=0)
        return self._edges

    @property
    def kdtree(self):
        if self._kdtree is None:
            self._kdtree = spatial.cKDTree(self.vertices)
        return self._kdtree

    @property
    def graph(self):
        if self._graph is None:
            self._graph = self.create_nx_graph()
        return self._graph

    def load_obj(self):
        # adapted from https://www.pygame.org/wiki/OBJFileLoader

        vertices = []
        faces = []
        normals = []

        for line in open(self.filename, "r"):
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

        self._faces = np.array(faces, dtype=np.int) - 1
        self._vertices = np.array(vertices, dtype=np.float)
        self._normals = np.array(normals, dtype=np.float)

    def load_h5(self):
        with h5py.File(self.filename, "r") as f:
            self._vertices = f["vertices"].value
            self._normals = f["normals"].value
            self._faces = f["faces"].value

    def write_h5(self):
        with h5py.File(self.filename, "w") as f:
            f.create_dataset("vertices", self.vertices, compression="gzip")
            f.create_dataset("faces", self.faces, compression="gzip")
            f.create_dataset("normals", self.normals, compression="gzip")

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

    def get_local_view(self, n_points, pc_align=False, center_node_id=None,
                       center_coord=None, method="kdtree", verbose=False):
        if center_node_id is None and center_coord is None:
            center_node_id = np.random.randint(len(self.vertices))

        if center_coord is None:
            center_coord = self.vertices[center_node_id]

        n_samples = np.min([n_points, len(self.vertices)])

        if method == "kdtree":
            dists, node_ids = self.kdtree.query(center_coord, n_samples)
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

        local_vertices = self.vertices[node_ids]

        if pc_align:
            local_vertices = self.calc_pc_align(local_vertices)
        return local_vertices, center_node_id

    def calc_pc_align(self, vertices):
        pca = decomposition.PCA(n_components=3)
        pca.fit(vertices)

        return pca.transform(vertices)

    def create_nx_graph(self):
        weights = np.linalg.norm(self.vertices[self.edges[:, 0]] - self.vertices[self.edges[:, 1]], axis=1)

        print(weights.shape)

        weighted_graph = nx.Graph()
        weighted_graph.add_edges_from(self.edges)

        for i_edge, edge in enumerate(self.edges):
            weighted_graph[edge[0]][edge[1]]['weight'] = weights[i_edge]

        return weighted_graph


