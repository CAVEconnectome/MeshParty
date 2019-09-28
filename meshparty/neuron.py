def calc_skeleton_overlap(pre_neuron, post_neuron, dbins):

    pass
    return pre_sk_lengths, post_sk_lengths

def calc_synapse_overlap(pre_neuron, post_neuron, dbins):

    pass
    return pre_sk_lengths, post_sk_lengths

class Neuron(object):
    """class for storing neurons

    Parameters
    ----------
    mesh : trimesh_io.Mesh
    skeleton : skeleton.Skeleton
    pre_syns: pd.DataFrame (of synapses)
    post_syns: pd.DataFrame (of synapses)
    annotations : dict
        dictionary with keys of strings and valeus of pd.DataFrame of annotations
        'synapses'
    """
    def __init__(self, mesh, skeleton, pre_syns=None, post_syns = None, annotations = None):
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


if __name___ == "__main__":

    def load_neuron(seg_id, dl, mesh_folder, sk_folder, cv_path):
        mm = trimesh_io.MeshMeta()

        mesh = trimesh_io.Mesh()
        sk = skeletonize.skeletonize_mesh(mesh1)
        sk = skeleton_io.read_skeleton_h5(mesh_folder + f"{seg_id}.h5")
        pre_syns = dl.query_synapses()
        post_syns = dl.query_synapses()
        return  Neuron(mesh1, sk1, pre_syns, post_syns)

    neurons = [load_neuron(id_) for id_ in neuron_ids]
    N = len(neurons)
    pre_neurons = []
    post_neurons = []
    for n in neurons:
        is_axon, qual = n.calc_axon_mask(.5)
        pre_neurons.append(n.apply_sk_mask(is_axon))
        post_neurons.append(n.apply_sk_mask(~is_axon))

    overlaps = np.zeros((N, N))
    for j, pre_n in enumerate(pre_neurons):
        for i, post_n in enumerate(post_neurons):
            overlaps[i,j]=calc_skeleton_overlap(pre_n, post_n)
            