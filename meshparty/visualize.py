import numpy as np
import plotly.graph_objs as go

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


def go_skeleton_forest(skf, color_by=None, **kwargs):
    data = []
    for sk in skf._skeletons:
        data.append(go_skeleton(sk, color_by=color_by, **kwargs))
    return data
    

def go_skeleton(sk, color_by=None, **kwargs):
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
  
    if color_by is None:
        color = 'rgb(125,55,255)'
    else:
        color = sk.vertex_properties[color_by]
  
    if 'mode' not in kwargs:
        mode='lines'
    else:
        mode = kwargs.pop('mode')
    
    if 'line' not in kwargs:
        line = dict(width=4)
    else:
        line = kwargs.pop('line')
    
    if 'color' not in line:
        line['color'] = color
        if color_by is not None:
            line['colorscale'] = 'Hot'
    go_skeleton = go.Scatter3d(x=xs, y=ys, z=zs, mode=mode, line=line, **kwargs)
    return go_skeleton

