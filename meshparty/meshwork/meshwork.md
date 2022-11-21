## Meshwork h5 file format description
The meshwork files contain a combination of meshes, skeletons, and associated data annotations (of particular interest are synapses). 

The file is an h5 file with various data blocks, mostly numpy arrays. The below describe the various blocks.

The file itself can be loaded into python classes, using meshwork_io.load_meshwork_mesh. 

The top level groups are
### mesh
The contains a mesh representation of the neuron
#### vertices
A Nx3 numpy array containing the vertices of the mesh in nanometers.
#### faces
A Kx3 numpy array containing the indices into the vertex array that represent triangular faces of the mesh
#### link_edges
A Lx2 numpy array containing indices into the vertex array that represent extra edges that link together some parts of the mesh that might be disconnected by just the faces.
#### mesh_mask
A N long boolean vector representing a masking state of the mesh vertices.  By default this is all true.
#### node_mask
A N long boolean vector representing a masking state of the mesh vertices.  By default this is all true. (redundant?)

### skeleton
#### vertices
A Mx3 numpy array that are a set of vertices that represent skeleton nodes
#### edges
A Kx2 set of edges that are indices into the vertices array. 
#### mesh_index
A M long vector that contains the index of the mesh vertex that each skeleton vertex corresponds to
#### mesh_to_skel_map
A N long vector that contains the skeleton vertex index that each mesh vertex maps to.  If the value = N then there is no mapping from that mesh vertex to the skeleton.
#### radius
A M long numpy vector that encodes an estimate of the radius of the mesh at this skeleton node 
#### root
A single index that represents the root of the skeleton.

### annotations
This group contains a set of keys that represent different annotation dataframes into the data.

Each dataframe contains various 
### version