import numpy as np
from operator import itemgetter
from megnet.data.graph import GaussianDistance, expand_1st
import threading


def to_list(x):
    """
    If x is not a list, convert it to list
    """
    if isinstance(x, list) or isinstance(x, tuple):
        return x
    else:
        return [x]


def index_rep_from_structure(structure, r=4):
    """
    Take a pymatgen structure and convert it to a index-type graph representation
    The graph will have node, distance, index1, index2, where node is a vector of Z number
    of atoms in the structure, index1 and index2 mark the atom indices forming the bond and separated by
    distance

    :param structure: (pymatgen structure)
    :param r: (float) distance cutoff
    :return: (dictionary)
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

    :param mp_ids: (list) ids of each struture
    :param graphs: (list of dictionary) list of graph dictionary for each structure
    :param targets: (list of float) correpsonding target values for each structure
    :return: tuple(node_features, edges_features, global_values, index1, index2, targets, and ids)
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


class ClassGenerator:
    """
    A generator class that assembles several structures (indicated by batch_size) and form (x, y) pairs for model training

    :param feature_list: (list of np.array) list of [1, N] atom features, where N is the atom number
    :param bond_list: (list of np.array) list of [1, M] distances, where M is the number of bonds
    :param global_list: (list of np.array) list of [1, G] state features
    :param index1_list: (list of integer) list of [1, M] one side atomic index of the bond
    :param index2_list: (list of integer) list of [1, M] the other side atomic index of the bond
    :param targets: (list of float) target values fr
    :param batch_size: (int) number of samples in a batch
    :param scaler: (object) transform the bond attributes on the fly. This helps to save tremendous space
        if a expanded distance is feature.
    :param centers: (list of float) Gausssian expansion center
    :param width: (float) width of Gaussian basis
    :param expand_class: (float) expansion class

    """
    def __init__(self,
                 feature_list,
                 bond_list,
                 global_list,
                 index1_list,
                 index2_list,
                 targets,
                 batch_size=128,
                 scaler=None,
                 centers=np.linspace(0, 5, 100),
                 width=0.5,
                 expand_class=GaussianDistance):
        """


        """
        self.feature_list = feature_list
        self.bond_list = bond_list
        self.global_list = global_list
        self.index1_list = index1_list
        self.index2_list = index2_list
        self.targets = targets
        self.batch_size = batch_size
        self.lock = threading.Lock()
        self.total_n = len(feature_list)
        self.max_step = int(np.ceil(self.total_n / batch_size))
        self.mol_index = list(range(self.total_n))
        self.mol_index = np.random.permutation(self.mol_index)
        self.i = 0
        self.featurizer = expand_class(centers=centers, width=width)
        self.scaler = scaler
        if self.scaler is None:
            self.scaler = _DummyTransformer()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            batch_index = self.mol_index[self.i * self.batch_size:(self.i + 1) * self.batch_size]

            it = itemgetter(*batch_index)
            feature_list_temp = it(self.feature_list)
            if self.batch_size == 1:
                feature_list_temp = (feature_list_temp,)
            gnode = []
            for i, j in enumerate(feature_list_temp):
                gnode += [i] * len(j)
            gbond = []
            bond_list_temp = it(self.bond_list)
            if self.batch_size == 1:
                bond_list_temp = (bond_list_temp,)

            for i, j in enumerate(bond_list_temp):
                gbond += [i] * len(j)
            feature_list_temp = np.concatenate(feature_list_temp, axis=0)
            bond_list_temp = self.scaler.transform(self.featurizer.convert(np.concatenate(bond_list_temp, axis=0)))
            global_list_temp = it(self.global_list)
            if self.batch_size == 1:
                global_list_temp = (global_list_temp,)
            global_list_temp = np.concatenate(global_list_temp, axis=0)
            index1_temp = it(self.index1_list)
            index2_temp = it(self.index2_list)
            if self.batch_size == 1:
                index1_temp = (index1_temp,)
                index2_temp = (index2_temp,)

            target_temp = it(self.targets)
            if self.batch_size == 1:
                target_temp = (target_temp,)
            target_temp = np.atleast_2d(target_temp)
            index1 = []
            index2 = []
            offset_ind = 0
            for ind1, ind2 in zip(index1_temp, index2_temp):
                index1 += [i + offset_ind for i in ind1]
                index2 += [i + offset_ind for i in ind2]
                offset_ind += (max(ind1) + 1)
            self.i += 1

            if self.i == self.max_step:
                self.i -= self.max_step
                self.mol_index = np.random.permutation(self.mol_index)

            return [expand_1st(feature_list_temp),
                    expand_1st(bond_list_temp),
                    expand_1st(global_list_temp),
                    expand_1st(index1),
                    expand_1st(index2),
                    expand_1st(gnode),
                    expand_1st(gbond)], \
                   expand_1st(target_temp)


class _DummyTransformer(object):
    """
    Does nothing
    """
    def transform(self, x):
        return x

