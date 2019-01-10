
import numpy as np
from meshparty import trimesh_io, trimesh_vtk, skeletonize
import plotly.graph_objs as go
import vtk

def go_xyz_block(xyz, **kwargs):
    '''
    Make a scatter3d plot from a Nx3 numpy.array
    '''
    if 'mode' not in kwargs:
        mode='markers'
    else:
        mode = kwargs.pop('mode')

    if 'marker' not in kwargs:
        marker=dict(color='rgb(0,0,0)',
                    size=3,
                    symbol='circle')
    else:
        marker=kwargs.pop('marker')

    return go.Scatter3d(x=xyz[:,0],
                        y=xyz[:,1],
                        z=xyz[:,2],
                        mode=mode,
                        marker=marker,
                        **kwargs)


def go_skeleton(sk, **kwargs):
    '''
    Make a skeleton graphics object for plotly.
    '''
    paths = sk.paths
    xs = []
    ys = []
    zs = []
    for path in paths:
        xs.extend(sk.vertices[path,0])
        xs.append(np.nan)
        ys.extend(sk.vertices[path,1])
        ys.append(np.nan)
        zs.extend(sk.vertices[path,2])
        zs.append(np.nan)

    if 'mode' not in kwargs:
        mode='lines'
    else:
        mode = kwargs.pop('mode')
    
    if 'line' not in kwargs:
        line=dict(color='rgb(125,55,255)',
                  width=2,
                  )
    else:
        line = kwargs.pop('line')
    
    go_skeleton = go.Scatter3d(x=xs, y=ys, z=zs, mode=mode, line=line, **kwargs)
    return go_skeleton

