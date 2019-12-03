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

The scale parameter is by default set to 4, for low resolution monitors this can help make higher resolution images, 
but will result in images that are larger than the specified width and height.  You'll have to experiment with your
computer to find what gives you the best rendering results. 

::

    trimesh_vtk.render_actors([mesh_actor],
                              filename='my_image.png'
                              do_save=True,
                              video_width=1600,
                              video_height=1200)

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
of Nx3 uint8 RGB or Nx4 RGBA values (0-255) so that you can color the mesh precisely as you'd like to.

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
        camera = trimesh_vtk.oriented_camera(
        [0,0,0], # focus point
        backoff=1000, # put camera 1000 units back from focus
        backoff_vector=[0, 0, -1], # back off in negative z
        up_vector = [0, -1, 0] # make up negative y
    )
    # make a mesh actor that is colored by this distance
    mesh_actor = trimesh_vtk.mesh_actor(mesh,
                                        vertex_colors=clrs,
                                        opacity=0.5)

    trimesh_vtk.render_actors([mesh_actor])

Movie making
------------
There are a number of functions in :mod:`meshparty.trimesh_vtk` that are designed to help you make movies,
as series of png images on disk.  

We'll review them from the simpliest to the most complex.  The first is :func:`meshparty.trimesh_vtk.render_actors_360`.
This function simply takes a list of actors and spins them around 360 degrees over a certain number of frames,
saving each view to disk.  Optionally it can take an initial camera, and it will always rotate the camera around 
whatever direction is up with respect to that camera.  You can use do_save=False, in order to see the movie render
without saving to disk.  Typically this happens faster than the saving to disk, and so you shouldn't trust the speed 
that the movie plays on your screen.  On a recent macbook pro, it plays about 3 times faster.

Here's an example, that assumes you already have a mesh_actor and a camera defined..

::

    from meshparty import trimesh_vtk

    trimesh_vtk.render_actors_360([mesh_actor],
                                  'movie_360',
                                  270,
                                  camera_start=camera,
                                  do_save=True)

If you want to move the camera in a more flexible fashion, then :func:`meshparty.trimesh_vtk.render_movie` is the
next most complex function to use.  This takes a list of integer keyframe times and a corresponding list of camera positions,
it will then render a movie for each frame, interpolating the camera between these keyframes.
You might notice that render_movie_360 is implemented by using render_movie.

Here's an example that simply uses two cameras with different zooms to create a zoom in effect.

::

    from meshparty import trimesh_vtk

    camera_1 = trimesh_vtk.oriented_camera(mesh.centroid, backoff=500)
    camera_2 = trimesh_vtk.oriented_camera(mesh.centroid, backoff=100)

    trimesh_vtk.render_movie([mesh_actor],
                             'movie_zoom',
                             [0,300],
                             [camera_1, camera_2],
                             do_save=True)

One feature to mention of :func:'meshparty.trimesh_vk.render_actors' is the return_keyframes parameter. If set to true, 
every time you press 'k' it will save the current camera parameters and return all the cameras in a list when you quit 
the interactive vtk window with the 'q' button.  This allows you to quickly and interactively setup camera keyframes,
to use in rendering video paths.

Finally, if you really want to dive into altering the visualization at each timepoint, there is 
:func:`meshparty.trimesh_vtk.render_movie_flexible`.  Rather than specifying a set of times and cameras,
this function allows you to pass a function (frame_change_function), which will be passed the actors, the camera, 
and the timepoint at each time point.  That function can modify the actors and/or the camera, and the resulting 
changes will be rendered.  You need to know some more detail about how to manipulate vtk objects to use this function 
effectively, but if you do it allows you to quickly prototype some powerful visualizations.  

As an example, here's how you might create a movie that slowly reveals an neuron over time as a function of how 
far away it is from one point on the mesh.

::

    from meshparty import trimesh_vtk
    from scipy import sparse
    import vtk
    ds = sparse.csgraph.dijkstra(mesh.csgraph,
                                    directed=False,
                                    indices=0)
        

    mesh_actor = trimesh_vtk.mesh_actor(mesh, color=(0,1,0), opacity=1.0)

    # set up one camera that is aimed at the starting vertex
    camera_start = trimesh_vtk.oriented_camera(mesh.vertices[0,:], backoff=20)

    # set up another that is aimed at the farthest vertex, but more zoomed out
    max_ind = np.argmax(ds * (~np.isinf(ds)))
    camera_end = trimesh_vtk.oriented_camera(mesh.vertices[max_ind,:], backoff=200)

    # make a camera interpolator with these two cameras
    max_frame = ds[max_ind]/(15000/30)
    camera_interp = trimesh_vtk.make_camera_interpolator([0, max_frame],
                                                         [camera_start, camera_end])

    def reveal_axon(actors, camera, t,
                    framerate=30, nm_per_sec=15000):

        nm_per_frame = nm_per_sec/framerate
        actor = actors[0]

        # set the opacity according to whether the vertex is close enough
        # given the time and the speed calculated
        opacity=(255*((ds/nm_per_frame)<t)).astype(np.uint8)

        # set the color to be green everywhere
        color = np.array([0,255,0], dtype=np.uint8)
        clr=color*np.ones((opacity.shape[0], 3), dtype=np.uint8)

        # concatenate these together to form one RGBA array
        c = np.hstack([clr, opacity[:,np.newaxis]])

        # convert that to vtk
        vtk_vert_colors = trimesh_vtk.numpy_to_vtk(c)
        vtk_vert_colors.SetName('colors')

        # set the actor to use this new coloring
        actor.GetMapper().GetInput().GetPointData().SetScalars(vtk_vert_colors)

        # tell vtk that you have updated this actor so it gets rendered.
        actor.GetMapper().GetInput().GetPointData().Modified()

        # use your interpolated camera to set the camera for this time point
        camera_interp.InterpolateCamera(t, camera)

    # use render_movie_flexible to call this function and render a movie
    trimesh_vtk.render_movie_flexible([mesh_actor],
                                      'reveal_axon_movie',
                                      np.arange(0,max_frame),
                                      reveal_axon,
                                      camera=camera_start,
                                      video_height=1080,
                                      video_width=1920,
                                      scale=1,
                                      do_save=True)

The result is a movie that should look like this, although of course it will depend on your mesh.

.. youtube:: a7IpaSNFbxU

Hopefully this demonstrates how you could arbitrarily alter the coloring
of a mesh over time.  You can also use vtk's transformation capabilities to move actors over time.

Encoding movies
---------------
trimesh_vtk does not have capacities for encoding png images into compressed movies.  
However, we would reccomend using moviepy for this task.

Below is a simple example for encoding a movie as an mp4 after installing moviepy. 

::

    import moviepy.editor as mpe
    clip = mpe.ImageSequenceClip('reveal_axon_movie',fps=30)
    clip.write_videofile('reveal_axon_movie.mp4')

moviepy <https://zulko.github.io/moviepy/> has some great documentation that tells you how you can add text, 
stitch clips together, perform fancy cross fade effects, and all sorts of fun things. 