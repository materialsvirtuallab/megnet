from operator import itemgetter
import numpy as np
import threading
from megnet.utils.general_utils import to_list, expand_1st
from monty.json import MSONable


class CrystalGraph(MSONable):
    def __init__(self, r=4):
        self.r = r

    def convert(self, structure, state_attributes=None):
        return structure2graph(structure, state_attributes, r=self.r)

    def __call__(self, structure, state_attributes=None):
        return self.convert(structure, state_attributes)


def itemgetter_list(l, indices):
    """
    Get indices of l and return a tuple
    :param l:  (list)
    :param indices: (list) indices
    :return: (tuple)
    """
    it = itemgetter(*indices)
    if np.size(indices) == 1:
        return it(l),
    else:
        return it(l)


def structure2graph(structure, state_attributes=None, r=4):
    """
    Take a pymatgen structure and convert it to a index-type graph representation
    The graph will have node, distance, index1, index2, where node is a vector of Z number
    of atoms in the structure, index1 and index2 mark the atom indices forming the bond and separated by
    distance
    :param structure: (pymatgen structure)
    :param state_attributes: (list) state attributes
    :param r: (float) distance cutoff
    :return: (dictionary)
    """
    atom_i_segment_id = []  # index list for the center atom i for all bonds (row index)
    atom_i_j_id = []  # index list for atom j
    atom_number = []
    all_neighbors = structure.get_all_neighbors(r, include_index=True)
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


def structure2input(structure, r=4, distance_convertor=None, **kwargs):
    """
    Take a pymatgen structure and convert it to a index-type graph representation as model input

    :param structure: (pymatgen structure)
    :param r: (float) cutoff radius
    :param state_attributes: (list) a list of state attributes
    :param distance_convertor: (object) convert numeric distance values into a vector as bond features
    :return: (dictionary) inputs for model.predict
    """
    graph = structure2graph(structure, r=r)
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


class GaussianDistance(MSONable):
    """
    Expand distance with Gaussian basis sit at centers and with width 0.5.

    Args
    :param centers: (np.array)
    :param width: (float)
    """
    def __init__(self, centers=np.linspace(0, 4, 20), width=0.5):
        self.centers = centers
        self.width = width

    def convert(self, d):
        """
        expand distance vector d with given parameters
        :param d: (1d array) distance array
        :return: (matrix) N*M matrix with N the length of d and M the length of centers
        """
        d = np.array(d)
        return np.exp(-(d[:, None] - self.centers[None, :]) ** 2. / self.width ** 2)


class ClassGenerator:
    """
    A generator class that assembles several structures (indicated by batch_size) and form (x, y) pairs for model training

    :param atom_features: (list of np.array) list of atom feature matrix,
    :param bond_features: (list of np.array) list of bond features matrix
    :param state_features: (list of np.array) list of [1, G] state features, where G is the global state feature dimension
    :param index1_list: (list of integer) list of (M, ) one side atomic index of the bond, M is different for different
        structures
    :param index2_list: (list of integer) list of (M, ) the other side atomic index of the bond, M is different for different
        structures, but it has to be the same as the correponding index1.
    :param targets: (numpy array), N*1, where N is the number of structures
    :param batch_size: (int) number of samples in a batch
    :param
    """

    def __init__(self,
                 atom_features,
                 bond_features,
                 state_features,
                 index1_list,
                 index2_list,
                 targets,
                 batch_size=128,
                 is_shuffle=True,
                 is_expand_distance=False,
                 distance_expansor=None,
                 **kwargs):
        self.atom_features = atom_features
        self.bond_features = bond_features
        self.state_features = state_features
        self.index1_list = index1_list
        self.index2_list = index2_list
        self.targets = targets
        self.batch_size = batch_size
        self.lock = threading.Lock()
        self.total_n = len(atom_features)
        self.max_step = int(np.ceil(self.total_n / batch_size))
        self.mol_index = list(range(self.total_n))
        self.mol_index = np.random.permutation(self.mol_index)
        self.i = 0
        self.is_shuffle = is_shuffle
        self.is_expand_distance = is_expand_distance
        self.distance_expansor = distance_expansor
        if self.is_expand_distance and self.distance_expansor is None:
            raise ValueError('Cannot expand distance with no expansor provided!')

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            batch_index = self.mol_index[self.i*self.batch_size:(self.i+1)*self.batch_size]

            it = itemgetter(*batch_index)
            feature_list_temp = itemgetter_list(self.atom_features, batch_index)

            gnode = []
            for i, j in enumerate(feature_list_temp):
                gnode += [i] * len(j)
            gbond = []
            connection_list_temp = itemgetter_list(self.bond_features, batch_index)

            for i, j in enumerate(connection_list_temp):
                gbond += [i] * len(j)
            feature_list_temp = np.concatenate(feature_list_temp, axis=0)
            connection_list_temp = np.concatenate(connection_list_temp, axis=0)
            if self.is_expand_distance:
                connection_list_temp = self.distance_expansor.convert(connection_list_temp)
            global_list_temp = itemgetter_list(self.state_features, batch_index)
            global_list_temp = np.concatenate(global_list_temp, axis=0)
            index1_temp = it(self.index1_list)
            index2_temp = it(self.index2_list)
            target_temp = it(self.targets)
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
                    expand_1st(connection_list_temp),
                    expand_1st(global_list_temp),
                    expand_1st(index1),
                    expand_1st(index2),
                    expand_1st(gnode),
                    expand_1st(gbond)], \
                    expand_1st(target_temp)

