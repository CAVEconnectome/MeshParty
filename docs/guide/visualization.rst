Visualization
=============

The module :mod:`meshparty.trimesh_vtk`  module assists users in visualizing meshes, skeletons, and points using
`vtk  <https://vtk.org/>`_

The module simplifies the complex vtk workflow to two basic steps.

    - the construction of a set of vtk.vtkActor's
    - the rendering of those actors

The module accomplishes this by reducing the set of possible interactions with vtk
and simplifying the interactions with objects stored in numpy arrays with uniform sized shapes.

Here's a simple example assuming you have a mesh,  using :func:`meshparty.trimesh_vtk.mesh_actor`
::

    from meshparty import trimesh_vtk

    mesh_actor = trimesh_vtk.mesh_actor(mesh,
                                        color=(1,0,0),
                                        opacity=0.5)
    trimesh_vtk.render_actors([mesh_actor])

This will pop up an interactive vtk window, though often it comes up behind your other windows, so look for it.

There are similar functions for making vtkActors from skeletons :func:`meshparty.trimesh_vtk.skeleton_actor`
or numpy arrays of points :func:`meshparty.trimesh_vtk.point_cloud_actor`,
:func:`meshparty.trimesh_vtk.linked_point_actor` 
If you would prefer to save an image to disk you can pass a filename and do_save.

::

    trimesh_vtk.render_actors([mesh_actor],
                              filename='my_image.png'
                              do_save=True,
                              VIDEO_WIDTH=1600,
                              VIDEO_HEIGHT=1200)

Camera Control
--------------

Often with large meshes you want to direct a camera to a particular point in space, zoom, and orientation.
Vtk makes this possible by specification with vtkCamera object.  We provide some simplified functions for specifying
cameras in a parameterization that makes more sense to us :func:`meshparty.trimesh_vtk.oriented_camera`.

::

    camera = trimesh_vtk.oriented_camera(
        [0,0,0], # focus point
        backoff=1000, # put camera 1000 units back from focus
        backoff_vector=[0, 0, -1], # back off in negative z
        up_vector = [0, -1, 0] # make up negative y
    )
    trimesh_vtk.render_actors([mesh_actor], camera=camera)

You might have a perspective that someone setup from a neuroglancer link.  You can convert that to a camera
using :func:`meshparty.trimesh_vtk.camera_from_ngl_state` with the state dictionary.

Advanced coloring
-----------------

One common visualization need is to color the visualized objects according to some data,
the actor creation functions provide some facilities to assist with this, again assuming 
the coloring data is in the form of numpy arrays.

vertex_colors will accept floating point values, in which case it will pass it through vtk's default colormap.
Often you want to have more explicit control over the colors of vertices, and so it will also accept a numpy array 
of Nx3 uint8 RGB values so that you can color the mesh precisely as you'd like to.

In order to help you create these colors, a colormapping function exists :func:`meshparty.trimesh_vtk.values_to_colors`.
You can use matplotlib or seaborn colormaps to help you create the coloring scheme you would like.

Here's an example which colors the mesh according to the distance from the first vertex of the mesh

::

    from meshparty import trimesh_vtk
    from scipy import sparse
    import seaborn as sns

    # measure the distance from the first vertex to all others
    # using dijkstra.
    ds = sparse.csgraph.dijkstra(mesh.csgraph,
                                 directed=False,
                                 indices=[0])
                                 
    # normalize values between 0 and 1
    color_data = ds/np.nanmax(ds)
    cmap = np.array(sns.color_palette('viridis', 1000))
    clrs = trimesh_vtk.values_to_colors(color_data, cmap)

    # make a mesh actor that is colored by this distance
    mesh_actor = trimesh_vtk.mesh_actor(mesh,
                                        vertex_colors=clrs,
                                        opacity=0.5)

    trimesh_vtk.render_actors([mesh_actor])

