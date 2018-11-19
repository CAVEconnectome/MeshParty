# DEPRECATED


# import numpy as np
# import h5py
# from scipy import spatial
# from sklearn import decomposition
# import plyfile
# import os
# import networkx as nx
#
# import cloudvolume
# from multiwrapper import multiprocessing_utils as mu
#
#
# def read_mesh_h5(filename):
#     """Reads a mesh's vertices, faces and normals from an hdf5 file"""
#     assert os.path.isfile(filename)
#
#     with h5py.File(filename) as f:
#         vertices = f["vertices"].value
#
#         if "faces" in f.keys():
#             faces = f["faces"].value
#         else:
#             faces = []
#
#         if "normals" in f.keys():
#             normals = f["normals"].value
#         else:
#             normals = []
#
#     return vertices, faces, normals
#
#
# def write_mesh_h5(filename, vertices, faces,
#                   normals=None, overwrite=False):
#     """Writes a mesh's vertices, faces (and normals) to an hdf5 file"""
#
#     if os.path.isfile(filename):
#         if overwrite:
#             os.remove(filename)
#         else:
#             raise Exception(f"File {filename} already exists")
#
#     with h5py.File(filename, "w") as f:
#         f.create_dataset("vertices", data=vertices, compression="gzip")
#         f.create_dataset("faces", data=faces, compression="gzip")
#
#         if normals is not None:
#             f.create_dataset("normals", data=normals, compression="gzip")
#
#
# def read_mesh_obj(filename):
#     """Reads a mesh's vertices, faces and normals from an obj file"""
#     vertices = []
#     faces = []
#     normals = []
#
#     for line in open(filename, "r"):
#         if line.startswith('#'): continue
#         values = line.split()
#         if not values: continue
#         if values[0] == 'v':
#             v = values[1:4]
#             vertices.append(v)
#         elif values[0] == 'vn':
#             v = map(float, values[1:4])
#             normals.append(v)
#         elif values[0] == 'f':
#             face = []
#             texcoords = []
#             norms = []
#             for v in values[1:]:
#                 w = v.split('/')
#                 face.append(int(w[0]))
#                 if len(w) >= 2 and len(w[1]) > 0:
#                     texcoords.append(int(w[1]))
#                 else:
#                     texcoords.append(0)
#                 if len(w) >= 3 and len(w[2]) > 0:
#                     norms.append(int(w[2]))
#                 else:
#                     norms.append(0)
#             faces.append(face)
#
#     vertices = np.array(vertices, dtype=np.float)
#     faces = np.array(faces, dtype=np.int) - 1
#     normals = np.array(normals, dtype=np.float)
#
#     return vertices, faces, normals
#
#
# def _download_meshes_thread(args):
#     """ Downloads meshes into target directory
#
#     :param args: list
#     """
#     seg_ids, cv_path, target_dir, fmt = args
#
#     cv = cloudvolume.CloudVolume(cv_path)
#     os.chdir(target_dir)
#
#     for seg_id in seg_ids:
#         if fmt == "hdf5":
#             mesh = cv.mesh.get(seg_id)
#             write_mesh_h5(f"{seg_id}.h5", mesh["vertices"], mesh["faces"])
#         elif fmt == "obj":
#             cv.mesh.save(seg_id)
#         else:
#             raise Exception(f"unknown fmt: {fmt}")
#
#
# def download_meshes(seg_ids, target_dir, cv_path, n_threads=1,
#                     verbose=False, fmt="obj"):
#     """ Downloads meshes in target directory (parallel)
#
#     :param seg_ids: list of ints
#     :param target_dir: str
#     :param cv_path: str
#     :param n_threads: int
#     :param fmt: str, desired file format ("obj" or "hdf5")
#     """
#
#     n_jobs = n_threads * 3
#     if len(seg_ids) < n_jobs:
#         n_jobs = len(seg_ids)
#
#     seg_id_blocks = np.array_split(seg_ids, n_jobs)
#
#     multi_args = []
#     for seg_id_block in seg_id_blocks:
#         multi_args.append([seg_id_block, cv_path, target_dir, fmt])
#
#     if n_jobs == 1:
#         mu.multiprocess_func(_download_meshes_thread,
#                              multi_args, debug=True,
#                              verbose=verbose, n_threads=n_threads)
#     else:
#        mu.multisubprocess_func(_download_meshes_thread,
#                                multi_args, n_threads=n_threads,
#                                package_name="meshparty")
#
#
# def refine_mesh():
#     pass
#
#
# class MeshMeta(object):
#     def __init__(self):
#         self.filename_dict = {}
#
#     def mesh(self, filename):
#         if not filename in self.filename_dict:
#             try:
#                 self.filename_dict[filename] = Mesh(filename)
#             except:
#                 self.filename_dict[filename] = None
#
#         return self.filename_dict[filename]
#
#
# class Mesh(object):
#     def __init__(self, filename):
#         self._vertices = []
#         self._normals = []
#         self._faces = []
#         self._filename = filename
#
#         self._kdtree = None
#         self._graph = None
#         self._edges = None
#
#         if not os.path.exists(filename):
#             raise Exception("File does not exist")
#
#         if filename.endswith(".obj"):
#             self.load_obj()
#         elif filename.endswith(".h5"):
#             self.load_h5()
#         else:
#             raise Exception("Unknown filetype")
#
#     @property
#     def filename(self):
#         return self._filename
#
#     @property
#     def vertices(self):
#         return self._vertices
#
#     @property
#     def size(self):
#         return len(self.vertices)
#
#     @property
#     def faces(self):
#         return self._faces
#
#     @property
#     def normals(self):
#         return self._normals
#
#     @property
#     def edges(self):
#         if self._edges is None:
#             self._edges = np.concatenate([self.faces[:, :2],
#                                           self.faces[:, 1:3]], axis=0)
#         return self._edges
#
#     @property
#     def kdtree(self):
#         if self._kdtree is None:
#             self._kdtree = spatial.cKDTree(self.vertices, balanced_tree=True)
#         return self._kdtree
#
#     @property
#     def graph(self):
#         if self._graph is None:
#             self._graph = self.create_nx_graph()
#         return self._graph
#
#     def load_obj(self):
#         """Reads data from an obj file"""
#         vs, fs, ns = read_mesh_obj(self.filename)
#
#         self._vertices = vs
#         self._faces = fs
#         self._normals = ns
#
#     def load_h5(self):
#         """Reads data from an hdf5 file"""
#         vs, fs, ns = read_mesh_h5(self.filename)
#
#         self._vertices = vs
#         self._faces = fs
#         self._normals = ns
#
#     def write_h5(self, overwrite=False):
#         """Writes data to an hdf5 file"""
#         normals = None if len(self.normals) == 0 else self.normals
#
#         write_mesh_h5(self.filename, self.vertices, self.faces,
#                       normals, overwrite=overwrite)
#
#     def write_vertices_ply(self, out_fname, coords=None):
#         """Writing vertex coordinates as a .ply file using plyfile"""
#
#         if coords is None:
#             coords = self.vertices
#
#         tweaked_array = np.array(list(zip(coords[:, 0], coords[:, 1], coords[:, 2])), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
#
#         vertex_element = plyfile.PlyElement.describe(tweaked_array, "vertex")
#
#         if not os.path.exists(os.path.dirname(out_fname)):
#             os.makedirs(os.path.dirname(out_fname))
#
#         plyfile.PlyData([vertex_element]).write(out_fname)
#
#     def get_local_view(self, n_points, pc_align=False, center_node_id=None,
#                        center_coord=None, method="kdtree", verbose=False,
#                        return_node_ids=False, svd_solver="auto", pc_norm=True):
#         if center_node_id is None and center_coord is None:
#             center_node_id = np.random.randint(len(self.vertices))
#
#         if center_coord is None:
#             center_coord = self.vertices[center_node_id]
#
#         n_samples = np.min([n_points, len(self.vertices)])
#
#         if method == "kdtree":
#             dists, node_ids = self.kdtree.query(center_coord, n_samples,
#                                                 n_jobs=-1)
#             if verbose:
#                 print(np.mean(dists), np.max(dists), np.min(dists))
#         elif method == "graph":
#            dist_dict = nx.single_source_dijkstra_path_length(self.graph,
#                                                              center_node_id,
#                                                              weight="weight")
#            sorting = np.argsort(np.array(list(dist_dict.values())))
#            node_ids = np.array(list(dist_dict.keys()))[sorting[:n_points]]
#         else:
#             raise Exception("unknown method")
#
#         local_vertices = self.vertices[node_ids].copy()
#
#         if pc_align:
#             local_vertices = self.calc_pc_align(local_vertices, svd_solver,
#                                                 pc_norm=pc_norm)
#
#         if return_node_ids:
#             return local_vertices, center_node_id, node_ids
#         else:
#             return local_vertices, center_node_id
#
#     def calc_pc_align(self, vertices, svd_solver, pc_norm=True):
#         if pc_norm:
#             vertices -= vertices.mean(axis=0)
#             vertices /= vertices.std(axis=0)
#         pca = decomposition.PCA(n_components=3, svd_solver=svd_solver,
#                                 copy=False)
#         return pca.fit_transform(vertices)
#
#     def create_nx_graph(self):
#         weights = np.linalg.norm(self.vertices[self.edges[:, 0]] -
#                                  self.vertices[self.edges[:, 1]], axis=1)
#
#         print(weights.shape)
#
#         weighted_graph = nx.Graph()
#         weighted_graph.add_edges_from(self.edges)
#
#         for i_edge, edge in enumerate(self.edges):
#             weighted_graph[edge[0]][edge[1]]['weight'] = weights[i_edge]
#
#         return weighted_graph
#
#
