Filtering
=========

A common need in mesh analysis is to reduce the number of vertices on the mesh by some criteria.

This is accomplished via a method on the :func:`meshparty.trimesh_io.Mesh.apply_mask`.

This returns a mesh which has filtered the mesh vertices according to the passed boolean mask.

However, although the class behaves as a smaller mesh, it still knows about the larger mesh index space it was created from.
It provides some useful functions for mapping indices between the masked space and the original unmasked space 
:func:`meshparty.trimesh_io.Mesh.map_indices_to_unmasked`,
:func:`meshparty.trimesh_io.Mesh.map_boolean_to_unmasked`, and the inverse 
:func:`meshparty.trimesh_io.Mesh.filter_unmasked_boolean`, 
:func:`meshparty.trimesh_io.Mesh.filter_unmasked_indices`

Filters
-------
the module :obj:`meshparty.mesh_filters` provides functions creating boolean masks, that can then be passed to apply_mask.

A simple example would be :func:`meshparty.mesh_filters.filter_largest_component` which uses connected component analysis
to figure out which vertices belong to the largest connected component and return a mask that only includes those vertices.
This will remove small interior components of a mesh, as well as any parts which are not properly linked.

::

    from meshparty import mesh_filters
    mask = mesh_filters.filter_largest_component(mesh)
    new_mesh = mesh.apply_mask(mask)


A more complex example would be :func:`meshparty.mesh_filters.filter_spatial_distance_from_points` that 
finds mesh vertices that are close to a set of one or more points.

More complex still, :func:`meshparty.mesh_filters.filter_two_point_distance` that finds mesh vertices
which are close to the shortest path between two points.
This is useful for filtering out stretch or dendrite of axon between two points.

