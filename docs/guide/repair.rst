Mesh Repair
===========

Mesh representations of objects can be imperfect.  Particular to neurons, meshes from one object can be artificially separated
into multiple connected components due to imperfections of the data because many segmentation approaches will only join together voxels
that share a connected component.  

For this reason, it can become necessary to add edges to the mesh that represent connections between distinct components.
The :mod:`meshparty.trimesh_repair` is oriented around this process.  Presently it downloads data from a remote server for 
a set of merge point operations that were performed on an object, and uses those locations to add edges between mesh vertices.

add_link_edges
--------------

The main routine of this module is :func:`meshparty.trimesh_repair.add_link_edges`.
This function downloads the merge points from a server, then maps those point to mesh points,
then goes through a process of finding the minimal set of edges between vertices which join 
distinct connected components that are near the merge locations.   These edges are what we refer to as link_edges. 

One can call add_link_edges through the method :func:`meshparty.trimesh_io.Mesh.add_link_edges`.
This will store the the edges separately from the other components of the mesh,
but a joint representation of the edges contained in faces and link_edges 
is available at :obj:`meshparty.trimesh_io.Mesh.graph_edges`.


Presently this functionality is designed to work with a specific deployment of the
PyChunkedGraph web-service. As such, :func:`meshparty.trimesh_repair.add_link_edges` 
expects a seg_id a dataset_name and a server_address.  

