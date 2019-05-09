import numpy as np
from operator import itemgetter
from megnet.utils.general_utils import to_list


def index_rep_from_structure(structure, r=4):
    """
    Take a pymatgen structure and convert it to a index-type graph representation
    The graph will have node, distance, index1, index2, where node is a vector
    of Z number of atoms in the structure, index1 and index2 mark the atom
    indices forming the bond and separated by distance

    Args:
        structure: (pymatgen structure)
        r: (float) distance cutoff

    Returns:
        (dictionary)
    """
    atom_i_segment_id = []  # index list for the center atom i for all bonds (row index)
    atom_i_j_id = []  # index list for atom j
    atom_number = []
    all_neighbors = structure.get_all_neighbors(r, include_index=True)
    distances = []

    for k, n in enumerate(all_neighbors):
        atom_number.append(structure.sites[k].specie.Z)
        if len(n) < 1:
            index = None
        else:
            _, distance, index = list(zip(*n))
            index = np.array(index)
            distance = np.array(distance)

        if index is not None:
            ind = np.argsort(index)
            it = itemgetter(*ind)
            index = it(index)
            index = to_list(index)
            index = [int(i) for i in index]
            distance = distance[ind]
            distances.append(distance)
            atom_i_segment_id.extend([k] * len(index))
            atom_i_j_id.extend(index)
        else:
            pass
    if len(distances) < 1:
        return None
    else:
        return {'distance': np.concatenate(distances),
                'index1': atom_i_segment_id,
                'index2': atom_i_j_id,
                'node': atom_number}


def graph_to_inputs(mp_ids, graphs, targets):
    """
    Expand the graph dictionary to form a list of features and targets

    Args:
        mp_ids: (list) ids of each struture
        graphs: (list of dictionary) list of graph dictionary for each structure
        targets: (list of float) correpsonding target values for each structure

    Returns:
         tuple(node_features, edges_features, global_values, index1, index2, targets, and ids)
    """
    nodes = []
    edges = []
    globs = []
    index1 = []
    index2 = []

    ids = []
    final_targets = []
    for id, g, t in zip(mp_ids, graphs, targets):
        if isinstance(g, dict):
            ids.append(id)
            nodes.append(np.array(g['node'], dtype='int32'))
            edges.append(g['distance'].ravel())
            globs.append([[0, 0]])
            index1.append(g['index1'])
            index2.append(g['index2'])
            final_targets.append([t])
    return nodes, edges, globs, index1, index2, final_targets, ids
