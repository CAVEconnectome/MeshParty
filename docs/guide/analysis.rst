Common Analytical Techniques
============================

Ray tracing
-----------

One calculation that can be useful in analyzing mesh representations, particularly of neurons, is to measure the 'thickness' of the mesh.
This is often referred to in computer vision research as the "Shape Diameter Function" or `SDF <https://link.springer.com/article/10.1007/s00371-007-0197-5>`_.  

This essentially measures the distance a ray travels before it hits the other side of the mesh.
Typically it will start at a vertex or face, be oriented along the vertex or face normal of the mesh, and be pointed inward.
The ray stops when it hits another face, and the distance is the SDF.

More complex procedures shine multiple rays in an small arc of directions centered around the normal and take the mean of rays.
Others will discard rays that hit the opposite side faces as oblique angles,
and only count rays that hit faces that are roughly oriented tangentially to the traced ray.

MeshParty has wrapped functionality that is part of trimesh to use pyembree to perform this SDF calculation. 
The method is presently available at :func:`meshparty.skeletonize.ray_trace_distance`.  

The skeletonize methods provide the option to save this value for all vertices along the skeleton in `sk.vertex_properties['rs']`. 

Smoothing
---------

Another calculation that comes up often is smoothing.  Either you want to smooth the locations of vertices on a mesh or skeleton, 
or you want to smooth a set of values at each vertex.  You want to smooth based upon the graph of the mesh or skeleton, so values 
change slowly across the surface of mesh, or along the length of the skeleton.  We have a general purpose function from smoothing 
(:func:`meshparty.skeletonize.smooth_graph`) that implements an iterative algorithm, where at each time step, values at the vertex 
are kept X% what they were before, and (1-X)% the average value of nearby vertices along the graph.  One can module the value of X, 
the number of iterations, and the how many edges you can cross and still be considered nearby. 

An example of smoothing the locations of vertices of a mesh using this function

::

    from meshparty import skeletonize
    new_verts = skeletonize.smooth_graph(mesh.vertices,
                                         mesh.graph_edges,
                                         neighborhood=1,
                                         r=.1,
                                         iterations=100)
    mesh.vertices = new_verts

An example of smoothing a value defined at each of the vertices

::

    from meshparty import skeletonize
    import numpy as np

    # make up some random values
    values = np.rand.rand(len(mesh.vertices))
    # smooth them on the mesh 
    new_vals = skeletonize.smooth_graph(values,
                                        mesh.graph_edges,
                                        neighborhood=1,
                                        r=.1,
                                        iterations=100)

The same function can equally be applied to a skeleton

::

    from meshparty import skeletonize

    new_sk_verts = skeletonize.smooth_graph(sk.vertices,
                                            sk.edges,
                                            neighborhood=2, 
                                            r=.1,
                                            iterations=100)


