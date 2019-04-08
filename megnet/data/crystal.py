from operator import itemgetter
import numpy as np
from megnet.utils.general_utils import expand_1st, to_list
from megnet.data.graph import GaussianDistance
from monty.json import MSONable


class CrystalGraph(MSONable):
    """
    Convert a crystal into a graph with z as atomic feature and distance as bond feature
    one can optionally include state features

    Args:
        r (float): cutoff radius
    """
    def __init__(self, r=4):
        self.r = r

    def convert(self, structure, state_attributes=None):
        """
        Take a pymatgen structure and convert it to a index-type graph representation
        The graph will have node, distance, index1, index2, where node is a vector of Z number
        of atoms in the structure, index1 and index2 mark the atom indices forming the bond and separated by
        distance
        :param structure: (pymatgen structure)
        :param state_attributes: (list) state attributes
        :return: (dictionary)
        """
        atom_i_segment_id = []  # index list for the center atom i for all bonds (row index)
        atom_i_j_id = []  # index list for atom j
        atom_number = []
        all_neighbors = structure.get_all_neighbors(self.r, include_index=True)
        distances = []
        state_attributes = state_attributes or [[0, 0]]
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
                    'node': atom_number,
                    'state': state_attributes}

    def get_input(self, structure, distance_convertor=None, **kwargs):
        """
        Take a pymatgen structure and convert it to a index-type graph
        representation as model input

        :param structure: (pymatgen structure)
        :param r: (float) cutoff radius
        :param state_attributes: (list) a list of state attributes
        :param distance_convertor: (object) convert numeric distance values
            into a vector as bond features
        :return: (dictionary) inputs for model.predict
        """
        graph = self.convert(structure)
        if distance_convertor is None:
            centers = kwargs.get('centers', np.linspace(0, 6, 100))
            width = kwargs.get('width', 0.5)
            distance_convertor = GaussianDistance(centers, width)

        gnode = [0] * len(structure)
        gbond = [0] * len(graph['index1'])

        return [expand_1st(graph['node']),
                expand_1st(distance_convertor.convert(graph['distance'])),
                expand_1st(np.array(graph['state'])),
                expand_1st(np.array(graph['index1'])),
                expand_1st(np.array(graph['index2'])),
                expand_1st(np.array(gnode)),
                expand_1st(np.array(gbond)),
                ]

    def __call__(self, structure, state_attributes=None):
        return self.convert(structure, state_attributes)


def graphs2inputs(graphs, targets):
    """
    Expand the graph dictionary to form a list of features and targets

    :param graphs: (list of dictionary) list of graph dictionary for each structure
    :param targets: (list of float) correpsonding target values for each structure
    :return: tuple(node_features, edges_features, global_values, index1, index2, targets)
    """
    nodes = []
    edges = []
    globs = []
    index1 = []
    index2 = []

    final_targets = []
    for g, t in zip(graphs, targets):
        if isinstance(g, dict):
            nodes.append(np.array(g['node'], dtype='int32'))
            edges.append(g['distance'].ravel())
            globs.append(g['state'])
            index1.append(g['index1'])
            index2.append(g['index2'])
            final_targets.append([t])
    return nodes, edges, globs, index1, index2, final_targets
