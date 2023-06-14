import collections
import numpy as np
import h5py
from scipy import spatial, sparse
from sklearn import decomposition
try:
    from pykdtree.kdtree import KDTree
except:
    KDTree = spatial.cKDTree
import os
import networkx as nx
import requests
import time
import re
from collections import defaultdict
import warnings
import logging
from functools import wraps
import cloudvolume
from cloudvolume.datasource.precomputed.mesh.multilod import ShardedMultiLevelPrecomputedMeshSource
from multiwrapper import multiprocessing_utils as mu

import trimesh
from trimesh import caching
try:
    from trimesh import exchange
except ImportError:
    from trimesh import io as exchange

from pymeshfix import _meshfix
from tqdm import trange
import DracoPy
from meshparty import utils, trimesh_repair

try:
    from caveclient import infoservice
    allow_framework_client = True
except ImportError:
    logging.warning(
        "Need to pip install caveclient to use dataset_name parameters")
    allow_framework_client = False


class EmptyMaskException(Exception):
    """Raised when applying a mask that has all zeros"""
    pass


def _get_cv_path_from_info(dataset_name, server_address=None, segmentation_type='graphene'):
    """Get the cloudvolume path from a dataset name. Segmentation type should be
       either `graphene` or `flat`.
    """
    if allow_framework_client is False:
        logging.warning(
            "Need to pip install caveclient to use dataset_name parameters")
        return None

    info = infoservice.InfoServiceClient(
        dataset_name=dataset_name, server_address=server_address)
    if segmentation_type == 'graphene':
        cv_path = info.graphene_source(format_for='cloudvolume')
    elif segmentation_type == 'flat':
        cv_path = info.flat_segmentation_source(format_for='cloudvolume')
    else:
        cv_path = None
    return cv_path


def read_mesh_h5(filename):
    """Reads a mesh's vertices, faces and normals from an hdf5 file
    assert's that this file exists.
    Will load normals, link_edges, and node_mask if they exist.

    Parameters
    ----------
    filename: str
        a path to a h5 file

    Returns
    -------
    :obj:`np.array`
        vertices, a Nx3 x,y,z coordinates (float)
    :obj:`np.array`
        faces, a Mx3 a,b,c index into vertices for triangle faces np.int32
    :obj:`np.array`:
        normals, A Mx3 x,y,z direction for face normals, np.float32
        [] if it doesn't exist
    :obj:`np.array`
        link_edges, a Kx2 a,b list of extra link edges to add to the mesh graph np.int32
        None if this doesn't exist
    :obj:`np.array`
        node_mask, a N length bool area of whether to mask this index (None if doesn't exist)
        None if this doesn't exist

    Raises
    ------
        AssertionError
            if the filename is not a file
    """
    assert os.path.isfile(filename)

    with h5py.File(filename, "r") as f:
        if "draco" in f.keys():
            mesh_object = DracoPy.decode_buffer_to_mesh(
                f["draco"][()].tostring())
            vertices = np.array(mesh_object.points).astype(np.float32)
            if len(vertices.shape) == 1:
                N = len(vertices)
                vertices = vertices.reshape((N//3, 3))
            faces = np.array(mesh_object.faces).astype(np.uint32)
        else:
            vertices = f["vertices"][()]
            faces = f["faces"][()]

        if len(faces.shape) == 1:
            faces = faces.reshape(-1, 3)

        if "normals" in f.keys():
            normals = f["normals"][()]
        else:
            normals = None

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
                  normals=None, link_edges=None, node_mask=None,
                  draco=False, overwrite=False):
    """Writes a mesh's vertices, faces (and normals) to an hdf5 file

    Parameters
    ----------
    filename: str
        a path to a h5 file to write a mesh
    vertices : np.array
        a Nx3 x,y,z coordinates (float)
    faces: np.array
        a Mx3 a,b,c index into vertices for triangle faces np.int32  
    normals: np.array
        a Mx3 x,y,z direction for face normals, np.float32
        if it doesn't exist (default None)
    link_edges: np.array
        a Kx2 a,b list of extra link edges to add to the mesh graph np.int32
        None if this doesn't exist (default None)
    node_mask: np.array
        a N length bool area of whether to mask this index (None if doesn't exist)
        None if this doesn't exist (default None)
    overwrite: False
        whether to overwrite the file, will return silently if mesh file exists already

    """

    if os.path.isfile(filename):
        if overwrite:
            os.remove(filename)
        else:
            return

    with h5py.File(filename, "w") as f:
        if draco:

            buf = DracoPy.encode_mesh_to_buffer(vertices.flatten('C'),
                                                faces.flatten('C'))
            f.create_dataset("draco", data=np.void(buf))
        else:
            f.create_dataset("vertices", data=vertices, compression="gzip")
            f.create_dataset("faces", data=faces, compression="gzip")

        if normals is not None:
            f.create_dataset("normals", data=normals, compression="gzip")

        if link_edges is not None:
            f.create_dataset("link_edges", data=link_edges, compression="gzip")

        if node_mask is not None:
            f.create_dataset("node_mask", data=node_mask, compression="gzip")


def read_mesh(filename):
    """Reads a mesh from obj or h5 file

    Parameters
    ----------
    filename: str
        a path to a obj or h5 file to read a mesh

    Returns
    -------
    :obj:`numpy.array`
        vertices, a Nx3 x,y,z coordinates (float)
    :obj:`numpy.array`
        faces, a Mx3 a,b,c index into vertices for triangle faces np.int32
    :obj:`numpy.array`:
        normals, A Mx3 x,y,z direction for face normals, np.float32
        None if it doesn't exist
    :obj:`numpy.array`
        link_edges, a Kx2 a,b list of extra link edges to add to the mesh graph np.int32
        None if this doesn't exist or is an obj file
    :obj:`numpy.array`
        node_mask, a N length bool area of whether to mask this index (None if doesn't exist)
        None if this doesn't exist or is an obj file

    """

    if filename.endswith(".obj"):
        with open(filename, 'r') as fp:
            mesh_d = exchange.obj.load_obj(fp)
        vertices = mesh_d['vertices']
        faces = mesh_d['faces']
        normals = mesh_d.get('normals', None)
        link_edges = None
        node_mask = None
    elif filename.endswith(".h5"):
        mesh_data = read_mesh_h5(filename)
        vertices, faces, normals, link_edges, node_mask = mesh_data
    else:
        raise Exception("Unknown filetype")
    return vertices, faces, normals, link_edges, node_mask


def _download_meshes_thread_graphene(args):
    """ Helper to Download meshes into target directory from graphene sources.
    Parameters
    ----------
    args : tuple
        seg_ids : iterator of ids
            the seg ids (with filenames = f"{seg_id}.h5") in the target_dir
        cv_path: str
            the cloudvolume path passed to cloudvolume.CloudVolume
        target_dir: str
            a path to the diretory to save the meshes
        fmt: str
            'hdf5', 'obj', 'stl' or any format supported by 'func':`write_to_file`
        overwrite: bool
            whether to overwrite the meshes if they already exist.
            will do no work if those don't exist
        merge_large_components: bool
            whether to merge all the large components using 'func':trimesh_io.Mesh.merge_large_components
            with default parameters (probably should be False)
        stitch_mesh_chunks: bool
            whether to stitch mesh chunks across meshes after downloading fragments (probably should be True)
        map_gs_to_https: bool
            whether to trigger cloudvolume.CloudVolume use_https option. Probably should be true unless you have
            a private bucket and have ~/.cloudvolume/secrets setup properly
        remove_duplicate_vertices: bool
            whether to bluntly merge duplicate vertices (probably should be False)
        chunk_size: tuple
            size of chunks when deduplicating
        save_draco: bool
            whether to save meshes as draco compressed
        progress: bool
            show progress bars

     """
    seg_ids, cv_path, target_dir, fmt, overwrite, \
        merge_large_components, stitch_mesh_chunks, map_gs_to_https, \
        remove_duplicate_vertices, progress, chunk_size, save_draco = args

    cv = cloudvolume.CloudVolume(cv_path, use_https=map_gs_to_https)

    for seg_id in seg_ids:
        print('downloading {}'.format(seg_id))
        target_file = os.path.join(target_dir, f"{seg_id}.h5")
        if not overwrite and os.path.exists(target_file):
            print('file exists {}'.format(target_file))
            continue
        print('file does not exist {}'.format(target_file))

        try:
            cv_mesh = cv.mesh.get(
                seg_id, remove_duplicate_vertices=remove_duplicate_vertices)[seg_id]

            faces = np.array(cv_mesh.faces)
            if len(faces.shape) == 1:
                faces = faces.reshape(-1, 3)

            mesh = Mesh(vertices=cv_mesh.vertices,
                        faces=faces,
                        process=False)

            if merge_large_components:
                mesh.merge_large_components()

            if fmt == "hdf5":
                write_mesh_h5(f"{target_dir}/{seg_id}.h5",
                              mesh.vertices,
                              mesh.faces.flatten(),
                              link_edges=mesh.link_edges,
                              draco=save_draco,
                              overwrite=overwrite)
            else:
                mesh.write_to_file(f"{target_dir}/{seg_id}.{fmt}")
        except Exception as e:
            print(e)


def _download_meshes_thread_precomputed(args):
    """ Helper to Download meshes into target directory

    Parameters
    ----------
    args : tuple
        seg_ids : iterator of ids
            the seg ids (with filenames = f"{seg_id}.h5") in the target_dir
        cv_path: str
            the cloudvolume path passed to cloudvolume.CloudVolume
        target_dir: str
            a path to the diretory to save the meshes
        fmt: str
            'hdf5', 'obj', 'stl' or any format supported by 'func':`write_to_file`
        overwrite: bool
            whether to overwrite the meshes if they already exist.
            will do no work if those don't exist
        merge_large_components: bool
            whether to merge all the large components using 'func':trimesh_io.Mesh.merge_large_components
            with default parameters (probably should be False)
        stitch_mesh_chunks: bool
            whether to stitch mesh chunks across meshes after downloading fragments (probably should be True)
        map_gs_to_https: bool
            whether to trigger cloudvolume.CloudVolume use_https option. Probably should be true unless you have
            a private bucket and have ~/.cloudvolume/secrets setup properly
        remove_duplicate_vertices: bool
            whether to bluntly merge duplicate vertices (probably should be False)
        chunk_size: tuple
            chuck size for deduplification
        progress: bool
            show progress bars
     """
    seg_ids, cv_path, target_dir, fmt, overwrite, \
        merge_large_components, stitch_mesh_chunks, \
        map_gs_to_https, remove_duplicate_vertices, \
        progress, chunk_size, save_draco = args

    cv = cloudvolume.CloudVolume(
        cv_path, use_https=map_gs_to_https,
        progress=progress,
    )

    download_segids = [
        segid for segid in seg_ids
        if overwrite or not os.path.exists(
            os.path.join(target_dir, f"{segid}.h5")
        )
    ]

    already_have = list(set(seg_ids).difference(set(download_segids)))

    if not overwrite:
        print("Already Have: " + str(already_have))

    print("Downloading: " + str(download_segids))

    while len(download_segids):
        download_now = download_segids[:100]
        download_segids = download_segids[len(download_now):]    
        if isinstance(cv.mesh, ShardedMultiLevelPrecomputedMeshSource):
            cv_meshes = cv.mesh.get(download_now)
        else:     
            cv_meshes = cv.mesh.get(
                download_now,
                remove_duplicate_vertices=remove_duplicate_vertices,
                fuse=False
            )

        for segid, cv_mesh in cv_meshes.items():
            mesh = Mesh(
                vertices=cv_mesh.vertices,
                faces=cv_mesh.faces,
                process=False,
            )

            if merge_large_components:
                mesh.merge_large_components()

            if fmt == "hdf5":
                write_mesh_h5(f"{target_dir}/{segid}.h5",
                              mesh.vertices,
                              mesh.faces.flatten(),
                              link_edges=mesh.link_edges,
                              draco=save_draco,
                              overwrite=overwrite)
            else:
                mesh.write_to_file(f"{target_dir}/{segid}.{fmt}")


def download_meshes(seg_ids, target_dir, cv_path, overwrite=True,
                    n_threads=1, verbose=False,
                    stitch_mesh_chunks=True,
                    merge_large_components=False,
                    remove_duplicate_vertices=False,
                    map_gs_to_https=True, fmt="hdf5",
                    save_draco=False,
                    chunk_size=None,
                    progress=False):
    """ Downloads meshes in target directory (in parallel)
    will break up the seg_ids into n_threads*3 job blocks or fewer and download them all

    Parameters
    ----------
    seg_ids : iterator of ids
            the seg ids (with filenames = f"{seg_id}.h5") in the target_dir
    target_dir: str
        a path to the diretory to save the meshes
    cv_path: str
        the cloudvolume path passed to cloudvolume.CloudVolume
    n_threads: int
        how many parallel processes to use when downloading (default 1)
    overwrite: bool
        whether to overwrite the meshes if they already exist.
        will do no work if those don't exist (default True)
    stitch_mesh_chunks: bool
        whether to stitch mesh chunks across meshes after downloading fragments (default True)
    merge_large_components: bool
        whether to merge all the large components using 'func':trimesh_io.Mesh.merge_large_components
        with default parameters (default False)
    remove_duplicate_vertices: bool
        whether to bluntly merge duplicate vertices (default False)
    map_gs_to_https: bool
        whether to trigger cloudvolume.CloudVolume use_https option. Probably should be true unless you have
        a private bucket and have ~/.cloudvolume/secrets setup properly (default True)
    chunk_size: np.array
        chunk size in nm to use in deduplification (default None)
    fmt: str
        'hdf5', 'obj', 'stl' or any format supported by :func:`meshparty.trimesh_io.Mesh.write_to_file` (default 'hdf5')
    progress: bool
    """

    if n_threads > 1:
        n_jobs = n_threads * 3
    else:
        n_jobs = 1

    if len(seg_ids) < n_jobs:
        n_jobs = len(seg_ids)

    # Use the cv path to establish if the source is graphene or not.
    if re.search('^graphene://', cv_path) is not None:
        _download_meshes_thread = _download_meshes_thread_graphene
    else:
        _download_meshes_thread = _download_meshes_thread_precomputed

    seg_id_blocks = np.array_split(seg_ids, n_jobs)

    multi_args = []
    for seg_id_block in seg_id_blocks:
        multi_args.append([seg_id_block, cv_path, target_dir, fmt,
                           overwrite, merge_large_components, stitch_mesh_chunks,
                           map_gs_to_https, remove_duplicate_vertices, progress, chunk_size, save_draco])

    if n_jobs == 1:
        mu.multiprocess_func(_download_meshes_thread,
                             multi_args, debug=True,
                             verbose=verbose, n_threads=n_threads)
    else:
        mu.multisubprocess_func(_download_meshes_thread,
                                multi_args, n_threads=n_threads,
                                package_name="meshparty", n_retries=40)


class MeshMeta(object):
    """ Manager class to keep meshes in memory and seemingless download them

        Parameters
        ----------
        cache_size: int
            number of meshes to keep in memory adapt this to your available memory and size of meshes
            set to zero to use less memory but read from disk cache
        cv_path: str
            path to pass to cloudvolume.CloudVolume
        dataset_name: str
            Dataset name to use to get cloudvolume path via infoservice
        server_address: str
            Server address for the infoservice. Uses a default value if None.
        segmentation_type: 'graphene' or 'flat'
            Selects which type of segmentation to use. Graphene is for proofreadable segmentations, flat is for static segmentations.
        disk_cache_path: str
            meshes are dumped to this directory => should be equal to target_dir
            in download_meshes (default None will not cache meshes)
        map_gs_to_https: bool
            whether to change gs paths to https paths, via cloudvolume's use_https option
        voxel_scaling: 3x1 numeric
            Allows a post-facto multiplicative scaling of vertex locations. These values are NOT saved, just used for analysis and visualization.
        """

    def __init__(self, cache_size=400, cv_path=None, dataset_name=None, server_address=None, segmentation_type='graphene',
                 disk_cache_path=None, map_gs_to_https=True, voxel_scaling=None):

        self._mesh_cache = {}
        self._cache_size = cache_size
        if cv_path is None and dataset_name is not None:
            cv_path = _get_cv_path_from_info(
                dataset_name=dataset_name, server_address=server_address, segmentation_type=segmentation_type)
        self._cv_path = cv_path
        self._cv = None
        self._map_gs_to_https = map_gs_to_https
        self._disk_cache_path = disk_cache_path
        self._voxel_scaling = voxel_scaling

        if self.disk_cache_path is not None:
            if not os.path.exists(self.disk_cache_path):
                os.makedirs(self.disk_cache_path)

    @property
    def cache_size(self):
        """the size of the cache"""
        return self._cache_size

    @property
    def cv_path(self):
        """str: the path passed to cloudvolume.CloudVolume"""
        return self._cv_path

    @property
    def disk_cache_path(self):
        """str: the path where meshes are saved"""
        return self._disk_cache_path

    @property
    def cv(self):
        """ cloudvoume.CloudVolume : the cloudvolume object"""
        if self._cv is None and self.cv_path is not None:
            self._cv = cloudvolume.CloudVolume(self.cv_path, parallel=10,
                                               use_https=self._map_gs_to_https)

        return self._cv

    @property
    def voxel_scaling(self):
        """np.array : 3 element vector to rescale mesh vertices"""
        return self._voxel_scaling

    def _filename(self, seg_id, lod=None):
        """ a method to define what path this seg_id will or is saved to

        Parameters
        ----------
        seg_id: np.uint64 or int
            the seg_id to get the filename for
        lod: int or None
            the level of detail oof the mesh

        """
        assert self.disk_cache_path is not None
        if lod is not None:
            return "%s/%d_%d.h5" % (self.disk_cache_path, seg_id, lod)
        else:
            return "%s/%d.h5" % (self.disk_cache_path, seg_id)

    def mesh(self, filename=None, seg_id=None, cache_mesh=True,
             merge_large_components=False,
             stitch_mesh_chunks=True,
             overwrite_merge_large_components=False,
             remove_duplicate_vertices=False,
             force_download=False,
             lod=0,
             voxel_scaling='default'):
        """ Loads mesh either from cache, disk or google storage

        Note, if the mesh is in a cache (memory or disk)
        you will get exactly what was in the cache
        irrespective of the other options you specified
        unless force_download is set, except voxel_scaling which is always applied post-facto.

        Parameters
        ----------
        filename: str
            the full path to a file to load (default None)
        seg_id: uint64
            the mesh_id to get (default None, requires cv_path)
        cache_mesh: bool
            if True: mesh is cached in a dictionary. The user is responsible
            for avoiding a memory overflow (default True)
        merge_large_components: bool
            if True: large (>100 vx) mesh connected components are linked
            and the additional edges strored in .link_edges
            this information is cached as well (default False)
        stitch_mesh_chunks: bool
            if True it will stitch the mesh fragments together into a single graph
            (default True)
        overwrite_merge_large_components: bool
            if True: recalculate large components (default False)
        remove_duplicate_vertices: bool
            whether to bluntly removed duplicate vertices (default False)
        lod: int
            what level of detail to download, only relevent for multi-resolution meshes (default =0 )
        force_download: bool
            whether to force the mesh to be redownloaded from cloudvolume
        voxel_scaling: 3 element numeric or None
            Allows a post-facto multiplicative scaling of vertex locations. These values are NOT saved, just used for analysis and visualization.
            By default, pulls from the value in the meshmeta. 

        Returns
        -------
        :obj:`Mesh`
            The mesh object of this seg_id 

        Raises
        ------
        AssertionError
            if filename is not None, and seg_id and cv_path are not both set
            then it doesn't know how to get your mesh
        """
        if not isinstance(self.cv.mesh, ShardedMultiLevelPrecomputedMeshSource):
            lod = None

        if voxel_scaling == 'default':
            voxel_scaling = self.voxel_scaling

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
                mesh.write_to_file(self._filename(seg_id, lod=lod))
        else:
            if self.disk_cache_path is not None and force_download is False:
                if os.path.exists(self._filename(seg_id, lod=lod)):
                    mesh = self.mesh(filename=self._filename(seg_id, lod=lod),
                                     cache_mesh=cache_mesh,
                                     merge_large_components=merge_large_components,
                                     overwrite_merge_large_components=overwrite_merge_large_components,
                                     voxel_scaling=voxel_scaling)
                    return mesh
            assert (seg_id is not None and self.cv is not None)
            if seg_id not in self._mesh_cache or force_download is True:
                
                if isinstance(self.cv.mesh, ShardedMultiLevelPrecomputedMeshSource):
                    cv_mesh_d = self.cv.mesh.get(seg_id, lod=lod)
                else:
                    cv_mesh_d = self.cv.mesh.get(
                        seg_id,  remove_duplicate_vertices=remove_duplicate_vertices)
                if isinstance(cv_mesh_d, (dict, collections.defaultdict)):
                    cv_mesh = cv_mesh_d[seg_id]
                else:
                    cv_mesh = cv_mesh_d
                faces = np.array(cv_mesh.faces)
                if (len(faces.shape) == 1):
                    faces = faces.reshape(-1, 3)

                mesh = Mesh(vertices=cv_mesh.vertices,
                            faces=faces)
                if isinstance(self.cv.mesh, ShardedMultiLevelPrecomputedMeshSource):
                    mesh=mesh.process()
                if cache_mesh and len(self._mesh_cache) < self.cache_size:
                    self._mesh_cache[seg_id] = mesh

                if self.disk_cache_path is not None:
                    mesh.write_to_file(self._filename(
                        seg_id, lod=lod), overwrite=force_download)
            else:
                mesh = self._mesh_cache[seg_id]

        mesh.voxel_scaling = voxel_scaling

        if (merge_large_components and (len(mesh.link_edges) == 0)) or \
                overwrite_merge_large_components:
            mesh.merge_large_components()
        return mesh


class Mesh(trimesh.Trimesh):
    """An extension of trimesh.Trimesh class to allow more features

    Parameters
    ----------
    *args : a list of :class:`trimesh.Trimesh` arguments
        the first most commonly used...
        vertices : np.array
            a Nx3 array of x,y,z of vertex positions
        faces: np.array
            a Mx3 array of abc indices into vertices that form triangle faces
        normals: np.array
            a Mx3 array of face normals
    node_mask : np.array
        a N long boolean array of which vertices are masked
    unmasked_size: np.array
        how long the original vertex list is (relevant for masked meshes)
    apply_mask: bool
        whether to apply the node_mask to the result
    link_edges: np.array
        a Kx2 array of indices into vertices that represent extra edges you 
        want to store in the mesh graph
    **kwargs:
        all the other keyword args you want to pass to :class:`trimesh.Trimesh`

    """

    def __init__(self, *args, node_mask=None, unmasked_size=None, apply_mask=False, link_edges=None, voxel_scaling=None, **kwargs):
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
            raise ValueError(
                'Original size cannot be smaller than current size')
        self._unmasked_size = unmasked_size

        if node_mask is None:
            node_mask = np.full(self.unmasked_size, True, dtype=bool)
        elif node_mask.dtype is not np.dtype('bool'):
            node_mask_inds = node_mask.copy()
            node_mask = np.full(self.unmasked_size, False, dtype=bool)
            node_mask[node_mask_inds] = True

        if len(node_mask) != unmasked_size:
            raise ValueError(
                'The node mask must be the same length as the unmasked size')

        self._node_mask = node_mask

        if apply_mask:
            if any(self.node_mask == False):
                nodes_f = vertices_all[self.node_mask]
                faces_f = utils.filter_shapes(
                    np.flatnonzero(node_mask), faces_all)[0]
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

        self._voxel_scaling = None
        self._MeshIndex = None

        super(Mesh, self).__init__(*new_args, **kwargs)
        if apply_mask:
            if link_edges is not None:
                if any(self.node_mask == False):
                    self.link_edges = utils.filter_shapes(
                        np.flatnonzero(node_mask), link_edges)[0]
                else:
                    self.link_edges = link_edges
            else:
                self.link_edges = None
        else:
            self.link_edges = link_edges

        self._index_map = None

        self.voxel_scaling = voxel_scaling

    # Helper class for handling scaling issues

    class ScalingManagement(object):
        @staticmethod
        def original_scaling(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                original_scaling = self.voxel_scaling
                self.voxel_scaling = None
                func(self, *args, **kwargs)
                self.voxel_scaling = original_scaling
            return wrapper

    @property
    def voxel_scaling(self):
        return self._voxel_scaling

    @voxel_scaling.setter
    def voxel_scaling(self, new_scaling):
        self._update_voxel_scaling(new_scaling)

    @property
    def inverse_voxel_scaling(self):
        if self.voxel_scaling is not None:
            return 1/self.voxel_scaling
        else:
            return None

    def _update_voxel_scaling(self, new_scaling):
        """Update the scale of the mesh

        Parameters
        ----------
        new_scale : 3-element vector 
            Sets the new xyz scale relative to the resolution from the mesh source
        """
        if self.voxel_scaling is not None:
            self.vertices = self.vertices * self.inverse_voxel_scaling

        if new_scaling is not None:
            self._voxel_scaling = np.array(new_scaling).reshape(3)
            self.vertices = self.vertices * self._voxel_scaling
        else:
            self._voxel_scaling = None

        self._clear_extra_cached_vertex_keys()

    def _clear_extra_cached_vertex_keys(self, keys=['nxgraph', 'csgraph', 'pykdtree', 'kdtree']):
        for k in keys:
            self._cache.delete(k)

    @property
    def link_edges(self):
        """numpy.array : a Kx2 set of extra edges you want to store in the mesh graph,
        :func:`edges` will return this plus :func:`face_edges`"""
        return self._data['link_edges']

    @link_edges.setter
    def link_edges(self, values):
        """this will invalidate the cached properties that are graph related"""
        if values is None:
            values = np.array([[], []]).T
        values = np.asanyarray(values, dtype=np.int64)
        # prevents cache from being invalidated
        with self._cache:
            self._data['link_edges'] = values
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
        """:class:`networkx.Graph` : networkx graph of the mesh"""
        return self._create_nxgraph()

    @caching.cache_decorator
    def csgraph(self):
        """:mod:`scipy.sparse.csgraph` : graph of the mesh"""
        return self._create_csgraph()

    @caching.cache_decorator
    def pykdtree(self):
        """pykdtree.KDTree : KDTree of the mesh vertices"""
        return KDTree(self.vertices)

    @caching.cache_decorator
    def kdtree(self, balanced_tree=False):
        """scipy.spatial.cKDTree : kdtree of the mesh vertices

        Parameters
        ----------
        balanced_tree: bool
            passed on to scipy.spatial.cKDTree
        """
        return spatial.cKDTree(self.vertices, balanced_tree=False)

    @property
    def n_vertices(self):
        """int : how many vertices are in the mesh"""
        return len(self.vertices)

    @property
    def n_faces(self):
        """int : how many faces are in the mesh"""
        return len(self.faces)

    @caching.cache_decorator
    def graph_edges(self):
        # mesh.edges has bidirectional edges, so we need to pass bidirectional link_edges.
        if len(self.link_edges) > 0:
            link_edges_sym = np.vstack(
                (self.link_edges, self.link_edges[:, [1, 0]]))
            link_edges_sym_unique = np.unique(link_edges_sym, axis=1)
        else:
            link_edges_sym_unique = self.link_edges
        return np.vstack([self.edges, link_edges_sym_unique])

    def fix_mesh(self, wiggle_vertices=False, verbose=False):
        """ Executes rudimentary fixing function from pymeshfix

        Good for closing holes, fixes mesh in place
        will recalculate normals

        Parameters
        ----------
        wiggle_vertices: bool
            adds robustness for smaller components (default False)
        verbose: bool
            whether to print out debug statements (default False)
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

        Parameters
        ----------
        n_points: int
            number of points to sample, default None will use the full mesh vertices
        max_dist: float
            sets an upper limit for distance of any sampled mesh point. Might
            reduce n_points
        sample_n_points: int
            has to be >= n_points; if > n_points more points are sampled and a
            subset randomly chosen
        fisheye: bool
            addition to sample_n_points; subset is sampled such that a fisheye
            effect is generated (default False)
        pc_align: bool
            computes PCA and orients mesh along PCs (default False)
        center_node_ids: list of ints
            mesh vertices at the center of the local views (default None)
        center_coords: list (n, 3) of floats
            coordinates at the center of the local views (default None)
            will override center_node_ids
            if both center_node_ids and center_coors are None will choose random spots
            equal to the length of vertices
        verbose: bool
            whether to print more debugging (default False)
        return_node_ids: bool
            sampled node ids are returned as well, changes the output format (default False)
        svd_solver: str
            PCA solver passed to sklearn.decomposition.PCA (default "auto")
        return_faces: bool
            sampled faces are returned as well, changes the output format (default False)
        adapt_unit_sphere_norm: bool
            NOT FUNCTIONAL (default False)
        pc_norm: bool
            if True: normalize point cloud to mean 0 and std 1 before PCA (default False)

        Returns
        -------
        np.array
            local_vertices, a list of n_points or n_sample_points x 3 matrix of points
            len(local_vertices)=K will depend on center_coords or center_coords_ind
        np.array
            center_node_ids, a K long array of center node ids.. useful if you had it choose random points
            if you had passed center_coords this will not accurately reflect the centers used
        np.array
            return_node_ids, Optional depending on whether return_node_ids. A K long list of 
        np.array
            return_faces, Optional depending on return_faces. faces on the local views, a K list of mx3 triangle faces. 

        """
        if center_node_ids is None and center_coords is None:
            center_node_ids = np.array([np.random.randint(len(self.vertices))])

        if center_coords is None:
            center_node_ids = np.array(center_node_ids, dtype=int)
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
                    ids = np.arange(0, sample_n_points, dtype=int)
                    for i_sample in range(len(center_coords)):
                        sample_ids = np.random.choice(ids, n_points,
                                                      replace=False,
                                                      p=probs[i_sample])
                        new_dists.append(dists[i_sample, sample_ids])
                        new_node_ids.append(node_ids[i_sample, sample_ids])

                    dists = np.array(new_dists, dtype=np.float32)
                    node_ids = np.array(new_node_ids, dtype=int)
                else:
                    ids = np.arange(0, sample_n_points, dtype=int)
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
                       adapt_unit_sphere_norm=False,
                       fisheye=False,
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
                                    adapt_unit_sphere_norm=adapt_unit_sphere_norm,
                                    fisheye=fisheye,
                                    verbose=verbose,
                                    return_node_ids=return_node_ids,
                                    svd_solver=svd_solver,
                                    return_faces=return_faces,
                                    pc_norm=pc_norm)

    def _filter_faces(self, node_ids):
        """ method to return reindexed faces that involve only certain vertices

        Parameters
        ----------
        node_ids: np.array
            a M long set of indices into vertices that you want to filter faces by
            so only return faces that involve these vertices. node_ids has to be sorted! 

        Returns
        -------
        np.array 
            a Kx3 matrix that is a proper faces for a mesh whose vertices = mesh.vertices[node_ids]
        """
        return utils.filter_shapes(node_ids, self.faces)

    def _filter_graph_edges(self, node_ids):
        """ method to return reindexed edges that involve only certain vertices

        Parameters
        ----------
        node_ids: np.array
            a M long set of indices into vertices that you want to filter graph_edges by
            so only return faces that involve these vertices. node_ids has to be sorted! 

        Returns
        -------
        np.array 
            a Kx2 matrix of edges for a mesh whose vertices = mesh.vertices[node_ids]
        """

        return utils.filter_shapes(node_ids, self.graph_edges)

    @ScalingManagement.original_scaling
    def add_link_edges(self, seg_id=None, merge_log=None, datastack_name=None, server_address=None,
                       close_map_distance=300, client=None, verbose=False, base_resolution=None):
        """ add a set of link edges to this mesh from a PyChunkedGraph endpoint
        This will ask the pcg server where merges were done and try to calculate 
        where edges should be added to reflect the merge operations that have been done
        on this mesh, linking disconnected portions of the mesh.

        Parameters
        ----------
        seg_id: int 
            the seg_id of this mesh
        merge_log : dict
            JSON dict of merge log as it comes out of the chunkedgraph client. If used, must
            also set base_resolution, the mip 0 resolution of the supervoxel segmentation volume.
        dataset_name: str or None, optional
            The datastack name this mesh can be found in. If None, requires a pre-made client
            passed through the client parameter. Defaults to None.
        close_map_distance: float, optional
            The distance in nm in mesh vertex coordinates to consider a mapping to be 'close'.
            Defaults to 300.
        server_address: str or None, optional
            the server address to find the pcg endpoint (defaults to None)
        client : annotationframeworkclient.FrameworkClient or None, optional
            Framework client for a specific datastack. If provided, ingores datastack name and
            server_address parameters. defaults to None
        verbose : bool, optional
            If True, provides more debugging statements, default is False
        base_resolution : array-like or None, optional
            Resolution of the supervoxel segmentation at its lowest mip.
        """
        if seg_id is None and merge_log is None:
            raise ValueError(
                'Must set either seg id or pre-determined merge log')

        if merge_log is not None:
            link_edges = trimesh_repair.merge_log_edges(self,
                                                        merge_log=merge_log,
                                                        base_resolution=base_resolution,
                                                        close_map_distance=close_map_distance,
                                                        verbose=verbose)
        else:
            # Use the get_link_edges approach
            link_edges = trimesh_repair.get_link_edges(self, seg_id, datastack_name=datastack_name,
                                                       close_map_distance=close_map_distance,
                                                       server_address=server_address,
                                                       verbose=verbose,
                                                       client=client)

        self.link_edges = np.vstack([self.link_edges, link_edges])

    def get_local_meshes(self, n_points, max_dist=np.inf, center_node_ids=None,
                         center_coords=None, pc_align=False, pc_norm=False,
                         fix_meshes=False):
        """ Extracts a local mesh

        Parameters
        ----------
        n_points: int
        max_dist: float
        enter_node_ids: list of ints
        center_coords: list (n, 3) of floats
        pc_align: bool
        pc_norm: bool
        fix_meshes: bool
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
        will add the edges to the existing set of link_edges
        or start a set of link_edges if there are None
        Note: can cause self-contacts to be innapropriately merged

        Parameters
        ----------
        size_threshold: int
            will merge components that have more than this many vertices
            (default 100)
        max_dist: float
            will only merge components that are closer than this
            (default 1000 in units of mesh.vertices, usually nm)
        dist_step: int
            will merge by marching in steps to look for things to merge
            this is the distance of each step (default 100 in units of mesh.vertices)

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
        """ Computes networkx graph for this mesh

        Returns
        -------
        :class:`networkx.Graph`
        """
        return utils.create_nxgraph(self.vertices, self.graph_edges, euclidean_weight=True,
                                    directed=False)

    def _create_csgraph(self):
        """ Computes scipy.sparse.csgraph with weights equal to euclidean distance
        with directed=False"""
        return utils.create_csgraph(self.vertices, self.graph_edges, euclidean_weight=True,
                                    directed=True)

    @property
    def node_mask(self):
        '''
        np.array: Returns the node/vertex mask currently applied to the data
        '''
        return self._node_mask

    @property
    def indices_unmasked(self):
        '''
        np.array: Gets the indices of nodes in the filtered mesh in the unmasked index array
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
        Makes a new Mesh by adding a new mask to the existing one.
        new_mask is a boolean array, either of the original vertex space or the
        current masked length (in which case it is padded with zeros appropriately).

        Parameters
        ----------
        new_mask: np.array
            a N long array of bool where False correponds to vertices that should be masked
            N needs to equal to mesh.vertices.shape[0] (or the original vertex shape if you are
            operating on an already masked mesh)
        kwargs: 
            keyword arguments to pass on to the new Mesh.__init__ function

        Returns
        -------
        trimesh_io.Mesh
            the mesh with the mask applied
        '''
        if not np.any(new_mask):
            raise(EmptyMaskException("new_mask is all False, mesh will be empty"))

        # We need to express the mask in the current vertex indices
        if np.size(new_mask) == np.size(self.node_mask):
            joint_mask = self.node_mask & new_mask
            new_mask = self.filter_unmasked_boolean(new_mask)
        elif np.size(new_mask) == self.vertices.shape[0]:
            joint_mask = self.node_mask & self.map_boolean_to_unmasked(
                new_mask)
        else:
            raise ValueError(
                'Incompatible shape. Must be either original length or current length of vertices.')

        if self.voxel_scaling is None:
            new_vertices = self.vertices
        else:
            new_vertices = self.vertices * self.inverse_voxel_scaling
        new_mesh = Mesh(new_vertices,
                        self.faces,
                        node_mask=joint_mask,
                        unmasked_size=self.unmasked_size,
                        voxel_scaling=self.voxel_scaling,
                        **kwargs)
        link_edge_unmask = self.map_indices_to_unmasked(self.link_edges)
        new_mesh._apply_new_mask_in_place(new_mask, link_edge_unmask)
        return new_mesh

    def _apply_new_mask_in_place(self, mask, link_edge_unmask):
        """ Internal function for applying masks.. use apply_mask
        Use builtin Trimesh tools for masking
        The new 0 index is the first nonzero element of the mask.
        Unfortunately, update_vertices maps all masked face values to 0 as well.
        """
        num_zero_expected = np.sum(
            self.faces == np.flatnonzero(mask)[0], axis=1)
        self.update_vertices(mask)

        num_zero_new = np.sum(self.faces == 0, axis=1)
        faces_to_keep = num_zero_new == num_zero_expected
        self.update_faces(faces_to_keep)
        self.link_edges = self.filter_unmasked_indices(link_edge_unmask)

    def map_indices_to_unmasked(self, unmapped_indices):
        '''
        For a set of masked indices, returns the corresponding unmasked indices

        Parameters
        ----------
        unmapped_indices: np.array
            a set of indices in the masked index space

        Returns
        -------
        np.array
            the indices mapped back to the original mesh index space
        '''
        return utils.map_indices_to_unmasked(self.indices_unmasked, unmapped_indices)

    def map_boolean_to_unmasked(self, unmapped_boolean):
        '''
        For a boolean index in the masked indices, returns the corresponding unmasked boolean index

        Parameters
        ----------
        unmapped_boolean : np.array
            a bool array in the masked index space

        Returns
        -------
        np.array
            a bool array in the original index space.  Is True if the unmapped_boolean suggests it should be.
        '''
        return utils.map_boolean_to_unmasked(self.unmasked_size, self.node_mask, unmapped_boolean)

    def filter_unmasked_boolean(self, unmasked_boolean):
        '''
        For an unmasked boolean slice, returns a boolean slice filtered to the masked mesh

        Parameters
        ----------
        unmasked_boolean : np.array
            a bool array in the original mesh index space

        Returns
        -------
        np.array
            returns the elements of unmasked_boolean that are still relevant in the masked index space
        '''
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

    @ScalingManagement.original_scaling
    def write_to_file(self, filename, overwrite=True, draco=False):
        """ Exports the mesh to any format supported by trimesh

        Parameters
        ----------
        filename: str
            the path to where to write the filename. Will use extension to infer format
            '.h5' for hdf5
            '.obj' for wavefront
            all others supported by :func:`trimesh.exchange.export.export_mesh`
        """
        if os.path.splitext(filename)[1] == '.h5':
            write_mesh_h5(filename,
                          self.vertices,
                          self.faces,
                          normals=self.face_normals,
                          link_edges=self.link_edges,
                          node_mask=self.node_mask,
                          draco=draco,
                          overwrite=overwrite)
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
