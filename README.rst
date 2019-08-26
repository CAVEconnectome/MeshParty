MeshParty
#########
docs https://meshparty.readthedocs.io/en/docs/index.html
 
############

From pypi:
::

    pip install meshparty


We try to keep the pypi version up to date. To install the git version do:

:: 

    git clone https://github.com/sdorkenw/MeshParty.git
    cd MeshParty
    pip install . --upgrade


to make optional features of ray tracing and interaction with the PyChunkedGraph work properly you need to install the optional dependancies

::

    conda install pyembree
    pip install annotationframeworkclient
    


Usage example
#################

::

    from mesh_party import trimesh_io

    meshmeta = trimesh_io.MeshMeta()
    mesh = meshmeta.mesh(path_to_mesh) # mesh gets cached

    local_vertices = mesh.get_local_view(n_points, pc_align=True, method="kdtree")


Downloading meshes
##################

Meshes can be downloaded in parallel using 

::

    trimesh_io.download_meshes(seg_ids, target_dir, cv_path)


where `cv_path` points to the cloudvolume bucket. For downloading proofread meshes one needs to 
specify the `mesh_endpoint` of the chunkedgraph server:

::

    trimesh_io.download_meshes(seg_ids, target_dir, cv_path, mesh_endpoint="https://...")



Extracting mesh information
###########################

The mesh needs to be `watertight` In order to compute reliable information. To
test whether a mesh is watertight, run

::

    mesh.is_watertight


To make a mesh watertight do
::

    mesh.fix_mesh()


Since trimesh_io.Mesh() inherits from trimesh.Trimesh all trimesh functionality 
is available to mesh, e.g.:
::

    mesh.volume
    mesh.area
    mesh.center_mass

