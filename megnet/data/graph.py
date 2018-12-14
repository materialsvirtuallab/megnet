from operator import itemgetter
import numpy as np
import threading


class GaussianDistance(object):
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

    :param feature_list: (list of np.array) list of atom feature matrix,
    :param bond_list: (list of np.array) list of bond features matrix
    :param global_list: (list of np.array) list of [1, G] state features, where G is the global state feature dimension
    :param index1_list: (list of integer) list of (M, ) one side atomic index of the bond, M is different for different
        structures
    :param index2_list: (list of integer) list of (M, ) the other side atomic index of the bond, M is different for different
        structures, but it has to be the same as the correponding index1.
    :param targets: (numpy array), N*1, where N is the number of structures
    :param batch_size: (int) number of samples in a batch
    """

    def __init__(self, feature_list, connection_list, global_list, index1_list, index2_list, targets, batch_size=128):
        self.feature_list = feature_list
        self.connection_list = connection_list
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

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            batch_index = self.mol_index[self.i*self.batch_size:(self.i+1)*self.batch_size]

            it = itemgetter(*batch_index)
            feature_list_temp = it(self.feature_list)
            gnode = []
            for i, j in enumerate(feature_list_temp):
                gnode += [i] * len(j)
            gbond = []
            connection_list_temp = it(self.connection_list)
            for i, j in enumerate(connection_list_temp):
                gbond += [i] * len(j)
            feature_list_temp = np.concatenate(feature_list_temp, axis=0)
            connection_list_temp = np.concatenate(connection_list_temp, axis=0)
            global_list_temp = np.concatenate(it(self.global_list), axis=0)
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


def expand_1st(x):
    """
    Adding an extra first dimension
    :param x: (np.array)
    :return: (np.array)
    """
    return np.expand_dims(x, axis=0)
