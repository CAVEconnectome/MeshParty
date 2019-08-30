Loading meshes
==============

MeshParty is designed to load mesh files from a variety of sources, including complete files on disk,
as well as from downloading meshes from a remote source via `CloudVolume 
<https://github.com/seung-lab/cloud-volume>`_ which supported a variety of formats,
but notably neuroglancer's `precomputed <https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed>`_,
`sharded <https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed#sharded-format>`_, and graphene. 

To facilitate downloading meshes there is a class :class:`meshparty.trimesh_io.MeshMeta`
which you pass an folder, and/or an cloudvolume path, and a memory cache size. 

Example
-------

Here's an example of downloading a mesh from the publicly available kasthuri2011 dataset.
(Kasthuri, Narayanan, et al. "Saturated reconstruction of a volume of neocortex." Cell 162.3 (2015): 648-661.)
::

    from meshparty import trimesh_io

    mm = trimesh_io.MeshMeta(
        cv_path = "precomputed://gs://neuroglancer-public-data/kasthuri2011/ground_truth",
        disk_cache_path = "test_meshes",
        map_gs_to_https=True)

    # load a segment
    mesh = mm.mesh(seg_id = 3710)

    # how many vertices and faces do we have
    print(mesh.vertices.shape, mesh.faces.shape)

You can also simply specify a path to an existing mesh on disk

::

    mesh = mm.mesh(filename = "path_to_my_mesh.obj")

Mesh
====

MeshMeta returns a :class:`meshparty.trimesh_io.Mesh` which is an extension of the :class:`trimesh.base.Trimesh` class.

This class provides a few extra properties and functions designed to assist large scale mesh analysis.
