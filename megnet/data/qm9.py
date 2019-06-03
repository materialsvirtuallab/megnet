"""
The data processing is based on the data shared by Faber, Felix A., et al.
"Prediction errors of molecular machine learning models lower than hybrid DFT error."
Journal of chemical theory and computation 13.11 (2017): 5255-5264..

Visit https://drive.google.com/open?id=0Bzn36Iqm8hZscHFJcVh5aC1mZFU. for access

"""
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from operator import itemgetter
from megnet.data.graph import GaussianDistance
import logging

atom_attri = ['type', 'chirality', 'ring_sizes', 'hybridization', 'acceptor',
              "donor", "aromatic"]
bond_attri = ['a_idx', 'b_idx', 'bond_type', "graph_distance", 'same_ring',
              'spatial_distance']
target_list = ['mu', 'alpha', 'HOMO', 'LUMO', 'gap', 'R2', 'ZPVE', 'U0', 'U',
               'H', 'G', 'Cv', 'omega1']

chem_accuracy = [0.1, 0.1, 0.043, 0.043, 1.2, 0.0012, 0.043, 0.05, 10]

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])


def load_qm9_faber(db_connection=None,
                   atom_attri=atom_attri,
                   bond_attri=bond_attri,
                   graph_dist=None,
                   restrict=None,
                   verbose=True):
    """
    Load the qm9 dataset from faber

    Args:
        db_connection: mongodb collection pointing to the qm9 data
        atom_attri: (list of string) atom attributes used as feature
        bond_attri: (list of string) bond attributes used as feature
        graph_dist: (list of integer) graph distances considered. Basically
        bonded atoms have graph distance of 1, second nearest neighbors are 2 etc.
        restrict: (dict) extra constraints for the query
        verbose: (bool) show the progress

    Returns:
        atom_features, bond_features, global_features, bond_atom_index1, bond_atom_index2, targets
    """
    # connection is necessary
    if db_connection is None:
        return None

    features_list = []
    connection_list = []
    global_list = []
    index1_list = []
    index2_list = []
    smiles = []
    qm9 = []
    targets = []
    # a_idx and b_idx are the index of atom connecting the bonds, they are not explicit features
    bond_attri = [i for i in bond_attri if i not in ['a_idx', 'b_idx']]
    # if no graph dist constraint is given, use all (max in data is 7)
    if graph_dist is None:
        graph_dist = list(range(10))
    if verbose:
        logging.info('Start querying...\n')
        index = 0

    for cur in db_connection.find():
        index += 1
        if verbose:
            if index % 1000 == 0:
                logging.info('%d-th molecule finished' % index)
        features = []
        connect = []
        index1 = []
        index2 = []
        smiles.append(cur['smiles'])
        qm9.append(cur['qm9'])
        targets.append(cur['mol_info'])

        total_weight = 0.0
        total_atom = 0.0
        for atoms in cur['atoms']:
            features.append([atoms[i] for i in atom_attri])
            total_weight += atoms['atomic_num']
            total_atom += 1

        total_bonds = 0.
        for bonds in cur['atom_pairs']:
            if bonds['graph_distance'] in graph_dist:
                index1.append(bonds['a_idx'])
                index2.append(bonds['b_idx'])
                connect.append([bonds[i] for i in bond_attri])
                total_bonds += 1
        # construct the complete bond info for all bonds
        # when 1 is bonded to 2, then 2 is bonded to 1
        index1 = index1 + index2
        index2 = index2 + index1
        connect += connect
        # sort the index according to index 1
        sort_index = np.argsort(index1)
        it = itemgetter(*sort_index)
        index1 = it(index1)
        index2 = it(index2)
        connect = it(connect)
        features_list.append(features)
        global_list.append([[total_weight / total_atom,
                             total_bonds / total_atom]])  # weight/atom, bond/atom
        connection_list.append(connect)
        index1_list.append(index1)
        index2_list.append(index2)
        if restrict is not None:
            if index > restrict:
                break
    targets = pd.DataFrame.from_records(targets)
    targets['smiles'] = smiles
    targets['qm9'] = qm9
    return features_list, connection_list, global_list, index1_list, index2_list, targets


def ring_to_vector(l):
    """
    Convert the ring sizes vector to a fixed length vector
    For example, l can be [3, 5, 5], meaning that the atom is involved
    in 1 3-sized ring and 2 5-sized ring. This function will convert it into
    [ 0, 0, 1, 0, 2, 0, 0, 0, 0, 0].

    Args:
        l: (list of integer) ring_sizes attributes

    Returns:
        (list of integer) fixed size list with the i-1 th element indicates number of
            i-sized ring this atom is involved in.
    """
    return_l = [0] * 9
    if l:
        for i in l:
            return_l[i - 1] += 1
    return return_l


class FeatureClean(BaseEstimator, TransformerMixin):
    """
    Clean the features and convert them to model inputs

    Args
        categorical: (list of string) variables that will be considered as categorical
        feature_labels: (list of string) all features
        distance_converter: (object) convert the spatial distance to expanded Gaussians
        is_norm_dist: (bool) whether to normalize the distance features
    """

    def __init__(self,
                 categorical=["type", 'chirality', 'hybridization', 'donor',
                              'acceptor', 'aromatic'],
                 feature_labels=['type', 'chirality', 'ring_sizes',
                                 'hybridization',
                                 'acceptor', "donor", 'aromatic'],
                 distance_converter=GaussianDistance(),
                 is_norm_dist=False):
        self.categorical = categorical
        self.feature_labels = feature_labels
        self.binarizer = {}
        self.scaler = StandardScaler()
        self.distance_converter = distance_converter
        self.c_labels = []
        self.has_numeric = None
        self.is_norm_dist = is_norm_dist

    def fit(self, X):
        """
        sklearn transformer interface

        Args:
            X: list of feature list, e.g., [[5, 0, [3, 5], 0, 0, 1, 1], [5, 0, [3, 5], 0, 0, 1, 1]] for the default
            feature labels
        Returns:
            (np.array) converted feature matrix
        """
        self.c_labels = []
        concated = np.concatenate([np.array(i, dtype=object) for i in X],
                                  axis=0)
        if concated.shape[1] != len(self.feature_labels):
            raise ValueError('Feature label dimension does not match data!')
        numerical_array = []

        for i, label in enumerate(self.feature_labels):
            # categorical features will be converted using LabelBinarizer in sklearn
            if label in self.categorical:
                column = list(concated[:, i])
                binarizer = LabelBinarizer()
                binarizer.fit(column)
                if len(binarizer.classes_) == 2:
                    n = 1
                else:
                    n = len(binarizer.classes_)
                self.c_labels.extend(['c'] * n)
                self.binarizer[label] = binarizer
            # ring_sizes feature will be converted using our diy converter
            elif label == 'ring_sizes':
                self.c_labels.extend(['c'] * 9)
            # spatial distance is converted by Gaussian expansion, note that this class works for
            # both atom and bond features
            elif label == 'spatial_distance':
                if self.is_norm_dist:
                    dist_symbol = 'n'
                else:
                    dist_symbol = 'c'
                self.c_labels.extend(
                    [dist_symbol] * self.distance_converter.centers.shape[-1])
                if dist_symbol == 'n':
                    numerical_array.append(self.distance_converter.convert(
                        np.array(concated[:, i], dtype=float)))
            else:
                self.c_labels.append('n')
                numerical_array.append(
                    np.array(concated[:, i], dtype=float)[:, None])
        # print([i.shape for i in numerical_array])
        if len(numerical_array) > 0:
            numerical_array = np.concatenate(numerical_array, axis=1)
            self.scaler.fit(numerical_array)
            self.has_numeric = True
        else:
            self.has_numeric = False
        return self

    def transform(self, X):
        """
        Transform new data according to learnt logics

        Args:
            X: list of features

        Returns:
            feature dimension
        """
        X_transformed = []
        for x in X:
            x_transformed = []
            x = np.array(x, dtype=object)
            for i, label in enumerate(self.feature_labels):
                column = x[:, i]
                if label in self.categorical:
                    x_transformed.append(
                        self.binarizer[label].transform(list(column)))
                elif label == 'ring_sizes':
                    ring_sizes = []
                    for l in column:
                        ring_sizes.append(ring_to_vector(l))
                    x_transformed.append(np.array(ring_sizes))

                elif label == 'spatial_distance':
                    x_transformed.append(self.distance_converter.convert(
                        np.array(column, dtype=float)))
                else:
                    x_transformed.append(np.array(column, dtype=float)[:, None])
                    # print([i.shape for i in x_transformed])
            concated = np.concatenate(x_transformed, axis=1)
            if self.has_numeric:
                n_columns = [i for i, j in enumerate(self.c_labels) if j == 'n']
                concated[:, n_columns] = self.scaler.transform(
                    concated[:, n_columns])
            X_transformed.append(concated)
        return X_transformed

    @staticmethod
    def find_index(label, labels):
        return [i for i, j in enumerate(labels) if j == label][0]


class Scaler(BaseEstimator, TransformerMixin):
    """
    Simple numerical scaler for list of features
    """

    def __init__(self):
        self.ss = StandardScaler()

    def fit(self, X):
        concated = np.concatenate([np.array(i) for i in X], axis=0)
        self.ss.fit(concated)
        return self

    def transform(self, X):
        X_transformed = []
        for x in X:
            X_transformed.append(self.ss.transform(x))
        return X_transformed


def sublist_from_qm9(ids, targets, *features):
    """
    select a subset of the data according to the qm9 id.

    Args:
        ids: (list of string) qm9 id list
        targets: (pandas DataFrame) data frame of targets
        features: (list）list of graph features

    Returns:
         subset of targets, features
    """

    def get_from_list(l, idx):
        it = itemgetter(*idx)
        return it(l)

    def id_from_qm9(qm9_id, targets):
        targets.loc[:, 'index'] = list(range(len(targets)))
        true_index = targets.set_index('qm9').loc[qm9_id].loc[:, 'index']
        return true_index.values

    true_index = id_from_qm9(ids, targets)
    new_features = []
    for f in features:
        new_features.append(get_from_list(f, true_index))
    return targets.set_index('qm9').loc[ids], new_features
