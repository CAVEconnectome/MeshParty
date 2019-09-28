import pickle
import pandas as pd
import numpy as np
from analysisdatalink.datalink_ext import AnalysisDataLinkExt as AnalysisDataLink
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from annotationframeworkclient.infoservice import InfoServiceClient
from meshparty import trimesh_io, trimesh_vtk, skeleton, skeletonize
from scipy import stats

def tolerance_filter(data, upper_thresh, lower_thresh, initial=False):
    """ hystertic threshold function
    
    runs a hysteretic threshold forward and backward over the data
    and returns the OR result of forwad and backward pass
    
    Parameters
    ----------
    data : np.array
        array of data to threshold
    upper_thresh : float
        upper threshold which data needs to cross to go True
    lower_thresh : float
        lower threshold which data needs to cross to go back False
    initial : bool
        whether to start False or True
    
    Returns
    -------
    thresh : np.bool
        whether data is high or low at each data point
    """
    

    hi = data >= upper_thresh
    lo_or_hi = (data <= lower_thresh) | hi
    ind = np.nonzero(lo_or_hi)[0]
    if not ind.size: # prevent index error if ind is empty
        return np.zeros_like(data, dtype=bool) | initial
    cnt = np.cumsum(lo_or_hi) # from 0 to len(x)
    return np.where(cnt, hi[ind[cnt-1]], initial)

def tolerance_filter_sym(data, upper_thresh, lower_thresh, initial=False):
    """ symettric hystertic threshold function
    
    runs a hysteretic threshold forward and backward over the data
    and returns the OR result of forwad and backward pass
    
    Parameters
    ----------
    data : np.array
        array of data to threshold
    upper_thresh : float
        upper threshold which data needs to cross to go True
    lower_thresh : float
        lower threshold which data needs to cross to go back False
    initial : bool
        whether to start False or True
    
    Returns
    -------
    thresh : np.bool
        whether data is high or low at each data point
    """
    for_hi = tolerance_filter(data, upper_thresh, lower_thresh, initial=initial)
    bac_hi = tolerance_filter(np.flip(data), upper_thresh, lower_thresh, initial=initial)
    bac_hi = np.flip(bac_hi)
    return for_hi | bac_hi

def calculate_cross_section_sk(mesh, sk, neigh=4000):
    """ function to calculate cross sectional area across skeleton of mesh
    
    Parameters
    ----------
    mesh : trimesh_io.Mesh
        a mesh to segment
    sk : skeleton.Skeleton
        skeletonization of the mesh, with K vertices
    neigh : float
        distance along skeleton to consider local when calculating
        local cross sectional area (default 4000)
    
    Returns
    -------
    cross_sections : np.array
        cross sectional area (in nm^2 if mesh.vertices in nm)
        
    """
    dm = sparse.csgraph.dijkstra(sk.csgraph,
                                 directed=False,
                                 limit = neigh)
    cross_sections=np.zeros(sk.n_vertices)
    dvs = sk.vertices[sk.edges[:, 0], :]-sk.vertices[sk.edges[:, 1], :]
    dvs = (dvs / np.linalg.norm(dvs, axis=1)[:, np.newaxis])
    for k, edge in enumerate(sk.edges):
        dv = dvs[k, :]
        vert_ind = edge[0]
        is_near_sk_vert = np.where(dm[vert_ind,:]<np.inf)[0]
    
        mesh_mask = np.isin(sk.mesh_to_skel_map, is_near_sk_vert)
        local_mesh = mesh.apply_mask(mesh_mask)
        sk_slice = local_mesh.section(plane_origin=sk.vertices[vert_ind,:], 
                                      plane_normal=dv)
        sk_2d, m = sk_slice.to_planar()
        cross_sections[vert_ind]= sk_2d.area
    return cross_sections
        

def baseline_als(y, lam, p, n_iter=10):
    """function for finding baseline in signal
    
    
    based upon paper:
    https://zanran_storage.s3.amazonaws.com/www.science.uva.nl/ContentPages/443199618.pdf
    
    implementation a minor adaptation from :
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    
    
    Parameters
    ----------
    y : np.array
        function to estimate baseline for
    lab : float
        lambda parameter (suggested 10-1000)
    p : float
        p parameter (suggesed .01 - .0001)
    n_iter : int
        number of iterations (default =10)
    
    Returns
    -------
    z : np.array
        baseline estimation of y
    """
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(n_iter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = sparse.linalg.spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def segment_boutons(mesh, sk, sk_metric,
                    lamb = 1000, p=.0001, n_iter=10,
                    high_thresh = 1.5, low_thresh=.5):
    """ function to segment boutons from an axon skeleton and mesh
    
    calculate a c
    iterates across the skeleton segments
    for each segment
        define a baseline using baseline_als
    Parameters
    ----------
    mesh : trimesh_io.Mesh
        a mesh to segment
    sk : skeleton.Skeleton
        skeletonization of the mesh, with K vertices
    sk_metric : np.array
        a float of 
    lamb : float
        the lambda parameter of baseline_als (default 1000)
    p : float
        the p parameters of baslines_als (default .0001)
    n_iter : int
        number of iterations for baselines_als (default 10)
    high_thresh : float
        the high threshold (in std) of cross sectional area
        hysteritic threshold
    low_thresh : float
        the low threshold (in std) of cross sectional area
        hysteritic threshold
        
    Returns
    -------
    is_bouton_sk_mask : np.array
        a K np.bool area of whether this sk.index is part of a bouton
    sk_metric_baseline : np.array
        a K array of baselines of metric calculated with baseline_als
    """
    is_bouton_sk_mask = np.zeros(sk.n_vertices, np.bool)
    metric_var=np.std(sk_metric)
    sk_metric_baseline = np.zeros(sk.n_vertices, np.float32)
    
    for k, seg in enumerate(sk.segments):
        seg_baseline=baseline_als(sk_metric[seg], lamb, p=p, n_iter=n_iter)
        seg_metric = sk_metric[seg]
        sk_metric_baseline[seg]=seg_baseline
        corrected_metric =seg_metric - seg_baseline
        scaled_metric = corrected_metric/metric_var
        is_bouton_seg = tolerance_filter_sym( scaled_metric, high_thresh, low_thresh)
        bouton_sk_inds = seg[is_bouton_seg]
        is_bouton_sk_mask[bouton_sk_inds]=True
    is_bouton_cell_mask = is_bouton_sk_mask[sk.mesh_to_skel_map]

    return is_bouton_sk_mask, sk_metric_baseline

def make_cross_sec_plot(mesh, sk, cross_sec, is_bouton_sk_mask, cross_sec_baseline,
                         ut=1.5, lt=.5, image_path=None, scale=1000):
    
    f, ax = plt.subplots(len(sk.segments),1, figsize = (12, 5*len(sk.segments)))

    for k, seg in enumerate(sk.segments):
        if len(sk.segments)>1:
            pa = ax[k]
        else:
            pa = ax
        bl = cross_sec_baseline[seg]
        pa.plot(sk.distance_to_root[seg]/scale, cross_sec[seg]/scale, label='corr_sc_crosssec')
        #pa.plot(sk.distance_to_root[seg], bl, label='baseline')
        #pa.plot(sk.distance_to_root[seg], bl + ut*np.std(cross_sec))
        #pa.plot(sk.distance_to_root[seg], bl + lt*np.std(cross_sec))
        is_b_t = is_bouton_sk_mask[seg]*(bl + ut*np.std(cross_sec))
        is_b_t = np.max(np.vstack([is_b_t,~is_bouton_sk_mask[seg]*bl]),axis=0)
        pa.plot(sk.distance_to_root[seg]/scale, is_b_t/scale)
    if image_path is not None:
        f.savefig(image_path)
        plt.close(f)