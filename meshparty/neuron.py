def calc_skeleton_overlap(neuron1, neuron2, dbins):

    pass
    return pre_sk_lengths, post_sk_lengths

def calc_synapse_overlap(neuron1, neuron2, dbins):

    pass
    return pre_sk_lengths, post_sk_lengths

class Neuron(object):
    """class for storing neurons

    Parameters
    ----------
    mesh : trimesh_io.Mesh
    skeleton : skeleton.Skeleton
    synapses: pd.DataFrame (of synapses)
    annotations : dict
        dictionary with keys of strings and valeus of pd.DataFrame of annotations
        'synapses'
    """
    def __init__(self, mesh, skeleton, synapses=None, annotations = None):
        self.mesh = mesh
        self.skeleton = skeleton
        self.synapses=synapses


    def calc_axon_mask(self, qual_threshold):
        """casey's splitting code here"""
        pass
        return is_axon, qual
    
    def apply_sk_mask(self, sk_mask):
        pass

    def apply_mesh_mask(self, mesh_mask):
        pass
