# MeshParty

## Installation
```
git clone https://github.com/sdorkenw/MeshParty.git
cd MeshParty
pip install . --upgrade
```

## Usage example

```
from mesh_party import mesh_io

meshmeta = mesh_io.MeshMeta()
mesh = meshmeta.mesh(path_to_mesh) # mesh gets cached

local_vertices = mesh.get_local_view(n_points, pc_align=True, method="kdtree")
```
