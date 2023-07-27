Skeletons
=========

Skeletonization
---------------
The :mod:`meshparty.skeletonize` module provides facilities to turn a mesh into a skeleton by implementing a Teasar [1]_ like algorithm
that is implemented on the mesh graph, meaning the graph of mesh vertices, where edges between vertices have weights according
to their euclidean distance. 

.. [1] Sato, M., Bitter, I., Bender, M. A., Kaufman, A. E., & Nakajima, M. (n.d.). TEASAR: tree-structure extraction algorithm for accurate and robust skeletons. In Proceedings the Eighth Pacific Conference on Computer Graphics and Applications. IEEE Comput. Soc. https://doi.org/10.1109/pccga.2000.883951

Example
-------
Again assuming you have a mesh object loaded, the key function is :func:`meshparty.skeletonize.skeletonize_mesh`

::

    from meshparty import skeletonize

    # skeletonize the mesh using a 12 um invalidation radius
    # assuming units of mesh.vertices is in nm
    sk = skeletonize_mesh(mesh, 
                          invalidation_d=12000)

There are a number of parameter options that are best understood after understanding the algorithm employed.

Algorithm
---------
The algorithm at it's core works on a connected component of the mesh graph.
Disconnected components are skeletonized separately, and trivially combined.

For each component, first a root node is found.  A discussion of root node finding is below.

Then a while loop is entered. Within the loop, first, the farthest still valid target along the mesh graph from the root node is found,
and the shortest path along the mesh is drawn from target to existing skeleton paths and added to the skeleton.

Second, vertices that are within the parameterized distance :obj:`invalidation_d` ALONG THE MESH GRAPH from that new skeleton path are invalidated.
If :obj:`compute_original_index` is selected, the algorithm will remember which skeleton path vertex was responsible for invalidating each mesh vertex.
This is saved in :obj:`skeleton.mesh_to_skel_map`.  Incidentally, the mesh index of every skeleton node index is saved at :obj:`skeleton.vertex_properties['mesh_index']`

This loop continues until all vertices are invalidated, and because we analyze one connected component at a time
this is guaranteed to finish.  Finally, at the end, optionally if :obj:`compute_radius` is selected, pyembree will be used to 
shine a ray along the vertex normal (pointed inward) of every skeleton node, and measure how far that ray travels till it 
intersects the other side of the mesh.  This is one way to estimate the local caliber of the mesh.
This is saved in :obj:`skeleton.vertex_properties['rs']`. 

------------
root finding
------------
In the case that a neuron is being skeletonized, there may be a soma for which this skeletonization routine is prone to 
produce too many minor branches along the surface.  To alleviate the problem optionally a soma location can be passed.
This position is the :obj:`soma_pt` parameter. 

This will cause the root to be the closest vertex in the mesh to be the soma point.
It will also pre-invalidate all vertices that are within a :obj:`soma_radius` of this point.
Finally, at the end it will optionally (:obj:`collapse_soma`) collapse all the skeleton nodes within that radius to be at the root. 

In the case where no soma point is passed, the algorithm with use a heuristic algorithm to choose a root.
It simply starts with a random vertex in the component, and then find the farther target from that index, 
then the farthest target from that target, and so on, until the next target is no longer any farther away
that the previous one. 

----------
advantages 
----------

The mesh is a vastly reduced representation of segmented objects compared to a voxelized segmentation. 
It is possible to store all the data from a single neuron in memory on a normal machine.  This means the algorithm
can be run within a global context of the neuron.  Voxelized skeletonization algorithms typically must break 
large data up into chunks, skeletonize each without any understanding of how that chunk fits into the global context,
and then hope to stitch the result of all those chunks back together again.  Typical mesh representations have already
separated data according to objects and so parallelism across objects is trivial, where voxelized approaches must pay 
a much larger IO and memory cost on every skeletonization approach. 
Dense skeletonization approaches such as `kimimaro <https://github.com/seung-lab/kimimaro>`_
effectively avoid these costs by skeletonizing all components in a chunk.
This however is not practical when segmentation is changing rapidly.

In addition, the result is directly tied to the mesh. In fact, skeleton vertices are guaranteed to be a subset of mesh vertices and there is a map between all mesh vertices
and the corresponding skeleton vertex which caused that vertex to be invalid.
This is useful analytically for correctly assigning say mesh nodes near synapses to skeleton nodes.

Finally, because the mesh graph can accurately reflect the true topology of the object.
Voxelized TEASAR approaches for example, typically use a spatial invalidation ball to roll down the path.
Axons or dendrites which are not connected to that path, but are nearby spatially can be inappropriately invalidated by such approaches.
By using the mesh graph to define distance, this kind of mistake can be avoided. 
A related point is locations where an object contacts itself.  
In neuroscience terms, when a dendrite touches another dendrite of the same cell,
or an axons of a cell touches its own dendrite.
Voxel based skeletonization often assumes that voxels that are adjacent are connected,
and thus cannot prevent skeletonization from crossing from axon to dendrite at such locations. 
The mesh graph can encode the fact the axon and dendrite come into contact but in fact there is no path
from one to the other at those self contact locations (assuming the mesh data is of high quality... see below)

-------------
disadvantages
-------------
The flip side of the algorithm having access to the mesh graph to more intelligently handle invalidation and self contacts,
is that it is sensitive to the validity of the mesh graph data.  It is commonplace for meshing approaches to produce
meshes which are perfectly reasonable for visualization, but not for this type of analysis.
For example, many mesh packages and processes remove duplicate vertices and re-index faces and edges
to reference unique coordinates.  This is a reasonable way to reduce the mesh and stitch together fragments
that might share faces.  However, as mentioned above when objects contact themselves, one doesn't always want to merge vertices.
On the other hand, there are also situations where meshes of objects can be disconnected, but in fact one wants them to be connected.
When axons get very small, and move at oblique angles, it is possible for voxels to not be connected.
In such case, many meshing approaches with produce a mesh which is disconnected, and this algorithm will skeletonize them separately.
There are potential ways to repair the mesh or the skeleton, but they conflict fundamentally with avoiding merging self contacts.
In summary, mesh based skeletonization requires a high quality mesh graph to be able to operate effectively. 

Voxelized skeletonization and traditional TEASAR like algorithms go to some efforts to keep skeletons in center of their objects.
This approach does not, an instead produces a skeleton path that lies on the outside of the mesh, not down its center.

If this is important to you, one can move skeleton vertices to be more in the center of objects by 
estimating the local caliber of the mesh (See :obj:`meshparty.skeleton.vertex_properties['rs']`) and then moving those vertices 
according to the vertex normal at those indices, and then smooth the result using :func:`meshparty.skeletonize.smooth_graph`.

Skeleton Analysis
-----------------
The returned skeleton objects are of :class:`meshparty.skeleton.Skeleton`, contain many of the same useful properties that meshes have.
Including :class:`networkx.Graph` and :mod:`scipy.sparse.csgraph` representations, as well breaking the skeleton into segments, finding tips. 

Skeleton IO
----------------
The :mod:`meshparty.skeleton_io` module has functions for reading and writing skeleton objects to disk as h5 files that preserve
all the data that have been calculated on these skeletons.