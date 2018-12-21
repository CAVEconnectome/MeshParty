__doc__ = """
Mesh Iterator Classes
"""

import random
import time
import numpy as np


ORDERS = ["random", "sequential"]


class LocalViewIterator(object):
    """
    Iterator class which samples local views that cover an entire mesh.
    Each mesh vertex is counted as "covered" if it's included in at
    least one local view across the iterator.
    """

    def __init__(self, mesh, n_points, batch_size=1, order="random",
                 pc_align=False, pc_norm=False, adaptnorm=False,
                 fisheye=False, sample_n_points=None, verbose=False):

        assert order in ORDERS, f"invalid order {order} not in {ORDERS}"

        self._active_inds = list(range(mesh.vertices.shape[0]))
        self._order = order
        self._mesh = mesh
        self._batch_size = batch_size

        # arguments for local view method calls
        self._kwargs = dict(n_points=n_points, pc_align=pc_align,
                            fisheye=fisheye,
                            verbose=verbose, sample_n_points=sample_n_points,
                            return_node_ids=True, pc_norm=pc_norm)

        self._deact_kwargs = dict(n_points=n_points, pc_align=pc_align,
                                  adapt_unit_sphere_norm=adaptnorm,
                                  verbose=verbose, sample_n_points=None,
                                  return_node_ids=True, pc_norm=pc_norm)

    def __iter__(self):
        return self

    def __next__(self):
        time_start = time.time()
        # stopping condition: no more indices to sample
        if len(self._active_inds) == 0:
            raise StopIteration

        n_samples = min(self._batch_size, len(self._active_inds))

        if self._order == "random":
            random.seed(time.time())
            centers = np.random.choice(self._active_inds, n_samples,
                                       replace=False)
            views, _, node_ids = self._mesh.get_local_views(
                center_node_ids=centers,  **self._kwargs)

            if self._kwargs["sample_n_points"] is not None:
                _, _, node_ids = self._mesh.get_local_views(
                    center_node_ids=centers, **self._deact_kwargs)

            self._deactivate_nodes(node_ids.flatten())
        elif self._order == "sequential":
            centers = []
            views = []
            while len(self._active_inds) > 0:
                centers.append(self._active_inds[0])
                view, _, node_ids = self._mesh.get_local_views(
                    center_node_ids=[centers[-1]], **self._kwargs)
                views.extend(view)

                if self._kwargs["sample_n_points"] is not None:
                    _, _, node_ids = self._mesh.get_local_views(
                        center_node_ids=centers, **self._deact_kwargs)

                self._deactivate_nodes(node_ids)
        else:
            raise Exception()

        print("Views took %.3fs" % (time.time() - time_start))
        return np.array(views, dtype=np.float32), \
               np.array(centers, dtype=np.uint32)

    def _deactivate_nodes(self, node_ids):
        """
        Removes nodes from consideration which have been sampled
        in the last patch
        """
        if isinstance(node_ids, np.ndarray):
            node_ids = node_ids.tolist()

        to_deactivate = set(node_ids)
        self._active_inds = list(filter(lambda x: x not in to_deactivate,
                                        self._active_inds))
