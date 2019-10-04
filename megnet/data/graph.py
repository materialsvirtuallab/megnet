"""Abstract classes and utility operations for building graph representations and
data loaders (known as Sequence objects in Keras).

Most users will not need to interact with this module."""
from abc import abstractmethod
from operator import itemgetter
import numpy as np
from megnet.utils.general import expand_1st, to_list
from megnet.utils.data import get_graphs_within_cutoff
from monty.json import MSONable
from megnet.data import local_env
from inspect import signature
from pymatgen.analysis.local_env import NearNeighbors
from keras.utils import Sequence


class StructureGraph(MSONable):
    """
    This is a base class for converting converting structure into graphs or model inputs
    Methods to be implemented are follows:
        1. convert(self, structure)
            This is to convert a structure into a graph dictionary
        2. get_input(self, structure)
            This method convert a structure directly to a model input
        3. get_flat_data(self, graphs, targets)
            This method process graphs and targets pairs and output model input list.

    """

    # TODO (wardlt): Consider making "num_*_features" funcs to simplify making a MEGNet model

    def __init__(self,
                 nn_strategy=None,
                 atom_converter=None,
                 bond_converter=None,
                 **kwargs):

        if isinstance(nn_strategy, str):
            strategy = local_env.get(nn_strategy)
            parameters = signature(strategy).parameters
            param_dict = {i: j.default for i, j in parameters.items()}
            for i, j in kwargs.items():
                if i in param_dict:
                    setattr(self, i, j)
                    param_dict.update({i: j})
            self.nn_strategy = strategy(**param_dict)
        elif isinstance(nn_strategy, NearNeighbors):
            self.nn_strategy = nn_strategy
        elif nn_strategy is None:
            self.nn_strategy = None
        else:
            raise RuntimeError("Strategy not valid")

        self.atom_converter = atom_converter
        self.bond_converter = bond_converter
        if self.atom_converter is None:
            self.atom_converter = self._get_dummy_converter()
        if self.bond_converter is None:
            self.bond_converter = self._get_dummy_converter()

    def convert(self, structure, state_attributes=None):
        """
        Take a pymatgen structure and convert it to a index-type graph representation
        The graph will have node, distance, index1, index2, where node is a vector of Z number
        of atoms in the structure, index1 and index2 mark the atom indices forming the bond and separated by
        distance.
        For state attributes, you can set structure.state = [[xx, xx]] beforehand or the algorithm would
        take default [[0, 0]]

        Args:
            state_attributes: (list) state attributes
            structure: (pymatgen structure)
            (dictionary)
        """
        state_attributes = state_attributes or [[0, 0]]
        index1 = []
        index2 = []
        bonds = []
        if self.nn_strategy is None:
            raise RuntimeError("NearNeighbor strategy is not provided!")
        for n, neighbors in enumerate(self.nn_strategy.get_all_nn_info(structure)):
            index1.extend([n] * len(neighbors))
            for neighbor in neighbors:
                index2.append(neighbor['site_index'])
                bonds.append(neighbor['weight'])

        atoms = [i.specie.Z for i in structure]

        if np.size(np.unique(index1)) < len(atoms):
            raise RuntimeError("Isolated atoms found in the structure")
        else:
            return {'atom': np.array(atoms, dtype='int32').tolist(),
                    'bond': bonds,
                    'state': state_attributes,
                    'index1': index1,
                    'index2': index2
                    }

    def __call__(self, structure):
        return self.convert(structure)

    def get_input(self, structure):
        """
        Turns a structure into model input
        """
        graph = self.convert(structure)
        return self.graph_to_input(graph)

    def graph_to_input(self, graph):
        """
        Turns a graph into model input

        Args:
            (dict): Dictionary description of the graph
        Return:
            ([np.ndarray]): Inputs in the form needed by MEGNet
        """
        gnode = [0] * len(graph['atom'])
        gbond = [0] * len(graph['index1'])

        return [expand_1st(self.atom_converter.convert(graph['atom'])),
                expand_1st(self.bond_converter.convert(graph['bond'])),
                expand_1st(np.array(graph['state'])),
                expand_1st(np.array(graph['index1'])),
                expand_1st(np.array(graph['index2'])),
                expand_1st(np.array(gnode)),
                expand_1st(np.array(gbond))]

    def get_flat_data(self, graphs, targets=None):
        """
        Expand the graph dictionary to form a list of features and targets tensors.
        This is useful when the model is trained on assembled graphs on the fly.

        Args:
            graphs: (list of dictionary) list of graph dictionary for each structure
            targets: (list of float or list) Optional: corresponding target
                values for each structure

        Returns:
            tuple(node_features, edges_features, global_values, index1, index2, targets)
        """

        output = []  # Will be a list of arrays

        # Convert the graphs to matrices
        for feature in ['atom', 'bond', 'state', 'index1', 'index2']:
            output.append([np.array(x[feature]) for x in graphs])

        # If needed, add the targets
        if targets is not None:
            output.append([to_list(t) for t in targets])

        return tuple(output)

    @staticmethod
    def _get_dummy_converter():
        return DummyConverter()

    def as_dict(self):
        all_dict = super().as_dict()
        if 'nn_strategy' in all_dict:
            nn_strategy = all_dict.pop('nn_strategy')
            all_dict.update({'nn_strategy': local_env.serialize(nn_strategy)})
        return all_dict

    @classmethod
    def from_dict(cls, d):
        if 'nn_strategy' in d:
            nn_strategy = d.pop('nn_strategy')
            nn_strategy_obj = local_env.deserialize(nn_strategy)
            d.update({'nn_strategy': nn_strategy_obj})
            return super().from_dict(d)
        return super().from_dict(d)


class StructureGraphFixedRadius(StructureGraph):
    """
    This one uses a short cut to call find_points_in_spheres cython function in
    pymatgen. It is orders of magnitude faster than previous implementations
    """

    def convert(self, structure, state_attributes=None):
        """
        Take a pymatgen structure and convert it to a index-type graph representation
        The graph will have node, distance, index1, index2, where node is a vector of Z number
        of atoms in the structure, index1 and index2 mark the atom indices forming the bond and separated by
        distance.
        For state attributes, you can set structure.state = [[xx, xx]] beforehand or the algorithm would
        take default [[0, 0]]

        Args:
            state_attributes: (list) state attributes
            structure: (pymatgen structure)
            (dictionary)
        """
        state_attributes = state_attributes or [[0, 0]]
        atoms = [i.specie.Z for i in structure]
        index1, index2, _, bonds = get_graphs_within_cutoff(structure, self.nn_strategy.cutoff)

        if np.size(np.unique(index1)) < len(atoms):
            raise RuntimeError("Isolated atoms found in the structure")
        else:
            return {'atom': np.array(atoms, dtype='int32').tolist(),
                    'bond': bonds,
                    'state': state_attributes,
                    'index1': index1,
                    'index2': index2
                    }

    @classmethod
    def from_structure_graph(cls, structure_graph):
        return cls(nn_strategy=structure_graph.nn_strategy,
                   atom_converter=structure_graph.atom_converter,
                   bond_converter=structure_graph.bond_converter)


class DistanceConverter(MSONable):
    """
    Base class for distance conversion. The class needs to have a convert method.
    """

    def convert(self, d):
        raise NotImplementedError


class DummyConverter(DistanceConverter):
    """
    Dummy converter as a placeholder
    """

    def convert(self, d):
        return d


class GaussianDistance(DistanceConverter):
    """
    Expand distance with Gaussian basis sit at centers and with width 0.5.

    Args:
        centers: (np.array)
        width: (float)
    """

    def __init__(self, centers=np.linspace(0, 5, 100), width=0.5):
        self.centers = centers
        self.width = width

    def convert(self, d):
        """
        expand distance vector d with given parameters

        Args:
            d: (1d array) distance array

        Returns
            (matrix) N*M matrix with N the length of d and M the length of centers
        """
        d = np.array(d)
        return np.exp(-(d[:, None] - self.centers[None, :]) ** 2 / self.width ** 2)


class MoorseLongRange(DistanceConverter):
    """
    This is an attempt to implement a Moorse/long range interactomic potential like
    distance expansion. The distance is expanded with this basis at different equilibrium
    bond distance, r_eq. It is still a work in progress. Do not use if you do not know
    much about the parameters
    ref: https://en.wikipedia.org/wiki/Morse/Long-range_potential#Function

    Args:
        d_e: (float) dissociate energy
        r_ref: (float) reference bond length
        r_eq: (list) equilibrium bond length
        p: (int) exponential term in the original equation, see ref
        q: (int) exponential term in the original equaiton, see ref
        cm: (list) long range coefficients in u_LR = \Sigma_i_N (cm_i / r^i)
        betas: (list) parameters determining the transition between long range and short range
    """

    def __init__(self, d_e=1, r_ref=2, r_eq=[1, 2, 3],
                 p=2, q=2, cm=[1, 2, 3, 4],
                 betas=[0.1, 0.2, 0.3, 0.4]):
        self.d_e = d_e
        self.r_ref = r_ref
        self.r_eq = np.array(r_eq)
        self.p = p
        self.q = q
        self.cm = np.array(cm).ravel()
        self.n_cm = len(self.cm)
        self.betas = np.array(betas).ravel()

    def convert(self, d):
        return self.d_e * (1 - self.u(d)[:, None] / self.u(self.r_eq)[None, :] *
                           np.exp(-self.beta(d) * self.y(d[:, None], self.r_eq[None, :], self.p))) ** 2

    def u(self, r):
        m_i = np.arange(1, self.n_cm + 1)
        if np.size(r) == 1:
            return np.sum(self.cm / r ** m_i)
        return np.sum(self.cm[None, :] / r[:, None] ** m_i[None, :], axis=1).ravel()

    @staticmethod
    def y(r, r_ref, p):
        return (r ** p - r_ref ** p) / (r ** p + r_ref ** p)

    def beta(self, r):
        y_p_ref = self.y(r, self.r_ref, self.p)
        y_q_ref = self.y(r, self.r_ref, self.q)
        return (self.beta_inf[None, :] * y_p_ref[:, None] + (1 - y_p_ref[:, None]) *
                np.sum(self.betas[None, :] * y_q_ref[:, None]
                       ** np.arange(0, len(self.betas))[None, :], axis=1).ravel()[:, None])

    @property
    def beta_inf(self):
        return np.log(2 * self.d_e / self.u(self.r_eq))


class BaseGraphBatchGenerator(Sequence):
    """Base class for classes that generate batches of training data for MEGNet.
    Based on the Sequence class, which is the data loader equivalent for Keras.

    Implementations of this base class must implement the :meth:`_generate_inputs`,
    which generates the lists of graph descriptions for a batch.

    The :meth:`process_atom_features` function and related functions are used to modify
    the features for each atom, bond, and global features when creating a batch.
    """

    def __init__(self, dataset_size, targets, batch_size=128, shuffle=True):
        """
        Args:
            dataset_size (int): Number of entries in dataset
            targets (ndarray): Feature to be predicted for each network
            batch_size (int): Maximum batch size
            shuffle (bool): Whether to shuffle the data after each step
        """
        if targets is not None:
            self.targets = np.array(targets)
        else:
            self.targets = None
        self.batch_size = batch_size
        self.total_n = dataset_size
        self.is_shuffle = shuffle
        self.max_step = int(np.ceil(self.total_n / batch_size))
        self.mol_index = np.arange(self.total_n)
        if self.is_shuffle:
            self.mol_index = np.random.permutation(self.mol_index)

    def __len__(self):
        return self.max_step

    def _combine_graph_data(self, feature_list_temp, connection_list_temp, global_list_temp,
                            index1_temp, index2_temp):
        """Compile the matrices describing each graph into single matrices for the entire graph

        Beyond concatenating the graph descriptions, this operation updates the indices of each
        node to be sequential across all graphs so they are not duplicated between graphs

        Args:
            feature_list_temp ([ndarray]): List of features for each node
            connection_list_temp ([ndarray]): List of features for each connection
            global_list_temp ([ndarray]): List of global state for each graph
            index1_temp ([ndarray]): List of indices for the start of each bond
            index2_temp ([ndarray]): List of indices for the end of each bond
        Returns:
            (tuple): Input arrays describing the entire batch of networks:
                - ndarray: Features for each node
                - ndarray: Features for each connection
                - ndarray: Global state for each graph
                - ndarray: Indices for the start of each bond
                - ndarray: Indices for the end of each bond
                - ndarray: Index of graph associated with each node
                - ndarray: Index of graph associated with each connection
        """
        # get atom's structure id
        gnode = []
        for i, j in enumerate(feature_list_temp):
            gnode += [i] * len(j)
        # get bond features from a batch of structures
        # get bond's structure id
        gbond = []
        for i, j in enumerate(connection_list_temp):
            gbond += [i] * len(j)

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
        index1 = []
        index2 = []
        offset_ind = 0
        for ind1, ind2 in zip(index1_temp, index2_temp):
            index1 += [i + offset_ind for i in ind1]
            index2 += [i + offset_ind for i in ind2]
            offset_ind += (max(ind1) + 1)

        # Compile the inputs in needed order
        inputs = [expand_1st(feature_list_temp),
                  expand_1st(connection_list_temp),
                  expand_1st(global_list_temp),
                  expand_1st(index1),
                  expand_1st(index2),
                  expand_1st(gnode),
                  expand_1st(gbond)]
        return inputs

    def on_epoch_end(self):
        if self.is_shuffle:
            self.mol_index = np.random.permutation(self.mol_index)

    def process_atom_feature(self, x):
        return x

    def process_bond_feature(self, x):
        return x

    def process_state_feature(self, x):
        return x

    def __getitem__(self, index):
        # Get the indices for this batch
        batch_index = self.mol_index[index * self.batch_size:(index + 1) * self.batch_size]

        # Get the inputs for each batch
        inputs = self._generate_inputs(batch_index)

        # Make the graph data
        inputs = self._combine_graph_data(*inputs)

        # Return the batch
        if self.targets is None:
            return inputs
        else:
            # get targets
            it = itemgetter(*batch_index)
            target_temp = itemgetter_list(self.targets, batch_index)
            target_temp = np.atleast_2d(target_temp)

            return inputs, expand_1st(target_temp)

    @abstractmethod
    def _generate_inputs(self, batch_index):
        """Get the graph descriptions for each batch

        Args:
             batch_index ([int]): List of indices for training batch
        Returns:
            (tuple): Input arrays describing each network:
                - [ndarray]: List of features for each node
                - [ndarray]: List of features for each connection
                - [ndarray]: List of global state for each graph
                - [ndarray]: List of indices for the start of each bond
                - [ndarray]: List of indices for the end of each bond
        """
        pass


class GraphBatchGenerator(BaseGraphBatchGenerator):
    """
    A generator class that assembles several structures (indicated by
    batch_size) and form (x, y) pairs for model training.

    Args:
        atom_features: (list of np.array) list of atom feature matrix,
        bond_features: (list of np.array) list of bond features matrix
        state_features: (list of np.array) list of [1, G] state features,
            where G is the global state feature dimension
        index1_list: (list of integer) list of (M, ) one side atomic index of the bond,
        M is different for different structures
        index2_list: (list of integer) list of (M, ) the other side atomic
            index of the bond, M is different for different structures,
            but it has to be the same as the corresponding index1.
        targets: (numpy array), N*1, where N is the number of structures
        batch_size: (int) number of samples in a batch
    """

    def __init__(self,
                 atom_features,
                 bond_features,
                 state_features,
                 index1_list,
                 index2_list,
                 targets=None,
                 batch_size=128,
                 is_shuffle=True):
        super().__init__(len(atom_features), targets, batch_size, is_shuffle)
        self.atom_features = atom_features
        self.bond_features = bond_features
        self.state_features = state_features
        self.index1_list = index1_list
        self.index2_list = index2_list

    def _generate_inputs(self, batch_index):
        """Get the graph descriptions for each batch

        Args:
             batch_index ([int]): List of indices for training batch
        Returns:
            (tuple): Input arrays describe each network:
                - [ndarray]: List of features for each nodes
                - [ndarray]: List of features for each connection
                - [ndarray]: List of global state for each graph
                - [ndarray]: List of indices for the start of each bond
                - [ndarray]: List of indices for the end of each bond
        """

        # Get the features and connectivity lists for this batch
        it = itemgetter(*batch_index)
        feature_list_temp = itemgetter_list(self.atom_features, batch_index)
        connection_list_temp = itemgetter_list(self.bond_features, batch_index)
        global_list_temp = itemgetter_list(self.state_features, batch_index)
        index1_temp = itemgetter_list(self.index1_list, batch_index)
        index2_temp = itemgetter_list(self.index2_list, batch_index)

        return feature_list_temp, connection_list_temp, global_list_temp, index1_temp, index2_temp


class GraphBatchDistanceConvert(GraphBatchGenerator):
    """
    Generate batch of structures with bond distance being expanded using a Expansor

    Args:
        atom_features: (list of np.array) list of atom feature matrix,
        bond_features: (list of np.array) list of bond features matrix
        state_features: (list of np.array) list of [1, G] state features, where G is the global state feature dimension
        index1_list: (list of integer) list of (M, ) one side atomic index of the bond, M is different for different
            structures
        index2_list: (list of integer) list of (M, ) the other side atomic index of the bond, M is different for
            different structures, but it has to be the same as the correponding index1.
        targets: (numpy array), N*1, where N is the number of structures
        batch_size: (int) number of samples in a batch
        is_shuffle: (bool) whether to shuffle the structure, default to True
        distance_converter: (bool) converter for processing the distances

    """

    def __init__(self,
                 atom_features,
                 bond_features,
                 state_features,
                 index1_list,
                 index2_list,
                 targets=None,
                 batch_size=128,
                 is_shuffle=True,
                 distance_converter=None):
        super().__init__(atom_features=atom_features,
                         bond_features=bond_features,
                         state_features=state_features,
                         index1_list=index1_list,
                         index2_list=index2_list,
                         targets=targets,
                         batch_size=batch_size,
                         is_shuffle=is_shuffle)
        self.distance_converter = distance_converter

    def process_bond_feature(self, x):
        return self.distance_converter.convert(x)


def itemgetter_list(l, indices):
    """
    Get indices of l and return a tuple

    Args:
        l:  (list)
        indices: (list) indices

    Returns:
        (tuple)
    """
    it = itemgetter(*indices)
    if np.size(indices) == 1:
        return it(l),
    else:
        return it(l)
