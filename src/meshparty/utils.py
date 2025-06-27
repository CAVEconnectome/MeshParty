import fastremap

import numpy as np
import pandas as pd


def remap_vertices_and_edges(
    id_list: np.ndarray,
    edgelist: np.ndarray,
):
    """Remap unique ids in a list to 0-N-1 range and remap edges accordingly."""

    id_map = {int(lid): ii for ii, lid in enumerate(id_list)}
    edgelist_new = fastremap.remap(
        edgelist,
        id_map,
    )
    return id_map, edgelist_new
