__doc__ = """
Mesh Iterator Classes
"""

import random
import time


ORDERS = ["random", "sequential"]


class LocalViewIterator(object):
    """
    Iterator class which samples local views that cover an entire mesh.
    Each mesh vertex is counted as "covered" if it's included in at
    least one local view across the iterator.
    """

    def __init__(self, mesh, n_points, order="random", pc_align=False,
                 pc_norm=True, method="kdtree", verbose=False):

        assert order in ORDERS, f"invalid order {order} not in {ORDERS}"

        self._active_inds = list(range(mesh.vertices.shape[0]))
        self._order = order
        self._mesh = mesh

        # arguments for local view method calls
        self._kwargs = dict(n_points=n_points, pc_align=pc_align,
                            method=method, verbose=verbose,
                            return_node_ids=True, pc_norm=pc_norm)

    def __iter__(self):
        return self

    def __next__(self):
        # stopping condition: no more indices to sample
        if len(self._active_inds) == 0:
            raise StopIteration

        if self._order == "random":
            random.seed(time.time())
            center = random.choice(self._active_inds)
        elif self._order == "sequential":
            center = self._active_inds[0]
        else:
            raise Exception()

        view, _, node_ids = self._mesh.get_local_view(center_node_id=center,
                                                      **self._kwargs)

        self._deactivate_nodes(node_ids)

        return view, center

    def _deactivate_nodes(self, node_ids):
        """
        Removes nodes from consideration which have been sampled
        in the last patch
        """

        to_deactivate = set(node_ids)
        self._active_inds = list(filter(lambda x: x not in to_deactivate,
                                        self._active_inds))
