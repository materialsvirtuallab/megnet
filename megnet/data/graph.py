from operator import itemgetter
import numpy as np
import threading
from megnet.utils.general_utils import expand_1st
from monty.json import MSONable


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


class DistanceConvertor(MSONable):
    """
    Base class for distance conversion. The class needs to have a convert method.
    """
    def convert(self, d):
        raise NotImplementedError


class GaussianDistance(DistanceConvertor):
    """
    Expand distance with Gaussian basis sit at centers and with width 0.5.

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
        return np.exp(-(d[:, None] - self.centers[None, :]) ** 2 / self.width ** 2)


class GraphBatchGenerator:
    """
    A generator class that assembles several structures (indicated by
    batch_size) and form (x, y) pairs for model training

    :param atom_features: (list of np.array) list of atom feature matrix,
    :param bond_features: (list of np.array) list of bond features matrix
    :param state_features: (list of np.array) list of [1, G] state features,
        where G is the global state feature dimension
    :param index1_list: (list of integer) list of (M, ) one side atomic index
        of the bond, M is different for different structures
    :param index2_list: (list of integer) list of (M, ) the other side atomic
        index of the bond, M is different for different structures, but it has
        to be the same as the correponding index1.
    :param targets: (numpy array), N*1, where N is the number of structures
    :param batch_size: (int) number of samples in a batch
    """
    def __init__(self,
                 atom_features,
                 bond_features,
                 state_features,
                 index1_list,
                 index2_list,
                 targets,
                 batch_size=128,
                 is_shuffle=True):
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

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            batch_index = self.mol_index[self.i * self.batch_size:(self.i + 1) * self.batch_size]
            it = itemgetter(*batch_index)
            # get atom features from  a batch of structures
            feature_list_temp = itemgetter_list(self.atom_features, batch_index)
            # get atom's structure id
            gnode = []
            for i, j in enumerate(feature_list_temp):
                gnode += [i] * len(j)

            # get bond features from a batch of structures
            connection_list_temp = itemgetter_list(self.bond_features, batch_index)
            # get bond's structure id
            gbond = []
            for i, j in enumerate(connection_list_temp):
                gbond += [i] * len(j)

            global_list_temp = itemgetter_list(self.state_features, batch_index)
            # assemble atom features together
            feature_list_temp = np.concatenate(feature_list_temp, axis=0)
            feature_list_temp = self.process_atom_feature(feature_list_temp)
            # assemble bond feature together
            connection_list_temp = np.concatenate(connection_list_temp, axis=0)
            connection_list_temp = self.process_bond_feature(connection_list_temp)
            # assemble state feature together
            global_list_temp = np.concatenate(global_list_temp, axis=0)
            global_list_temp = self.process_state_feature(global_list_temp)

            # assemble bond indices
            index1_temp = it(self.index1_list)
            index2_temp = it(self.index2_list)
            index1 = []
            index2 = []
            offset_ind = 0
            for ind1, ind2 in zip(index1_temp, index2_temp):
                index1 += [i + offset_ind for i in ind1]
                index2 += [i + offset_ind for i in ind2]
                offset_ind += (max(ind1) + 1)
            # get targets
            target_temp = it(self.targets)
            target_temp = np.atleast_2d(target_temp)
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
                    expand_1st(gbond)], expand_1st(target_temp)

    def process_atom_feature(self, x):
        return x

    def process_bond_feature(self, x):
        return x

    def process_state_feature(self, x):
        return x


class GraphBatchDistanceConvert(GraphBatchGenerator):
    """
    Generate batch of structures with bond distance being expanded using a Expansor

    :param atom_features: (list of np.array) list of atom feature matrix,
    :param bond_features: (list of np.array) list of bond features matrix
    :param state_features: (list of np.array) list of [1, G] state features, where G is the global state feature dimension
    :param index1_list: (list of integer) list of (M, ) one side atomic index of the bond, M is different for different
        structures
    :param index2_list: (list of integer) list of (M, ) the other side atomic index of the bond, M is different for different
        structures, but it has to be the same as the correponding index1.
    :param targets: (numpy array), N*1, where N is the number of structures
    :param batch_size: (int) number of samples in a batch
    :param is_shuffle: (bool) whether to shuffle the structure, default to True
    :param distance_convertor: (bool) convertor for processing the distances

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
                 distance_convertor=None):
        super(GraphBatchDistanceConvert, self).__init__(atom_features=atom_features,
                                                        bond_features=bond_features,
                                                        state_features=state_features,
                                                        index1_list=index1_list,
                                                        index2_list=index2_list,
                                                        targets=targets,
                                                        batch_size=batch_size,
                                                        is_shuffle=is_shuffle,
                                                        )
        self.distance_convertor = distance_convertor

    def process_bond_feature(self, x):
        return self.distance_convertor.convert(x)
