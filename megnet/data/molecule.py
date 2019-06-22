"""
Tools for creating graph inputs from molecule data
"""

import os
import sys
import itertools
from typing import List
from functools import partial
from collections import deque
from multiprocessing import Pool

import numpy as np
from pymatgen import Molecule, Element
from pymatgen.io.babel import BabelMolAdaptor

from megnet.data.qm9 import ring_to_vector
from megnet.utils.general import fast_label_binarize
from megnet.data.graph import (StructureGraph, GaussianDistance,
                               BaseGraphBatchGenerator, GraphBatchGenerator)

try:
    import pybel
except ImportError:
    pybel = None

try:
    from rdkit import Chem
except ImportError:
    Chem = None

__date__ = '12/01/2018'

# List of features to use by default for each atom
_ATOM_FEATURES = ['element', 'chirality', 'formal_charge', 'ring_sizes',
                  'hybridization', 'donor', 'acceptor', 'aromatic']

# List of features to use by default for each bond
_BOND_FEATURES = ['bond_type', 'same_ring', 'spatial_distance', 'graph_distance']

# List of elements in library to use by default
_ELEMENTS = ['H', 'C', 'N', 'O', 'F']


class SimpleMolGraph(StructureGraph):
    """
    Default using all atom pairs as bonds. The distance between atoms are used
    as bond features. By default the distance is expanded using a Gaussian
    expansion with centers at np.linspace(0, 4, 20) and width of 0.5
    """
    def __init__(self,
                 nn_strategy='AllAtomPairs',
                 atom_converter=None,
                 bond_converter=None
                 ):
        if bond_converter is None:
            bond_converter = GaussianDistance(np.linspace(0, 4, 20), 0.5)
        super().__init__(nn_strategy=nn_strategy, atom_converter=atom_converter,
                         bond_converter=bond_converter)


class MolecularGraph(StructureGraph):
    """Class for generating the graph inputs from a molecule

    Computes many different features for the atoms and bonds in a molecule, and prepares them
    in a form compatible with MEGNet models. The :meth:`convert` method takes a OpenBabel molecule
    and, besides computing features, also encodes them in a form compatible with machine learning.
    Namely, the `convert` method one-hot encodes categorical variables and concatenates
    the atomic features

    ## Atomic Features

    This class can compute the following features for each atom

    - `atomic_num`: The atomic number
    - `element`: (categorical) Element identity. (Unlike `atomic_num`, element is one-hot-encoded)
    - `chirality`: (categorical) R, S, or not a Chiral center (one-hot encoded).
    - `formal_charge`: Formal charge of the atom
    - `ring_sizes`: For rings with 9 or fewer atoms, how many unique rings
    of each size include this atom
    - `hybridization`: (categorical) Hybridization of atom: sp, sp2, sp3, sq.
    planer, trig, octahedral, or hydrogen
    - `donor`: (boolean) Whether the atom is a hydrogen bond donor
    - `acceptor`: (boolean) Whether the atom is a hydrogen bond acceptor
    - `aromatic`: (boolean) Whether the atom is part of an aromatic system

    ## Atom Pair Features

    The class also computes features for each pair of atoms

    - `bond_type`: (categorical) Whether the pair are unbonded, or in a single, double, triple, or aromatic bond
    - `same_ring`: (boolean) Whether the atoms are in the same aromatic ring
    - `graph_distance`: Distance of shortest path between atoms on the bonding graph
    - `spatial_distance`: Euclidean distance between the atoms. By default, this distance is expanded into
        a vector of 20 different values computed using the `GaussianDistance` converter

    """
    def __init__(self, atom_features=None, bond_features=None, distance_converter=None,
                 known_elements=None):
        """
        Args:
            atom_features ([str]): List of atom features to compute
            bond_features ([str]): List of bond features to compute
            distance_converter (DistanceCovertor): Tool used to expand distances
                from a single scalar vector to an array of values
            known_elements ([str]): List of elements expected to be in dataset. Used only if the
                feature `element` is used to describe each atom
        """

        # Check if openbabel and RDKit are installed
        if Chem is None or pybel is None:
            raise RuntimeError('RDKit and openbabel must be installed')

        super().__init__()
        if bond_features is None:
            bond_features = _BOND_FEATURES
        if atom_features is None:
            atom_features = _ATOM_FEATURES
        if distance_converter is None:
            distance_converter = GaussianDistance(np.linspace(0, 4, 20), 0.5)
        if known_elements is None:
            known_elements = _ELEMENTS

        # Check if all feature names are valid
        if any(i not in _ATOM_FEATURES for i in atom_features):
            bad_features = set(atom_features).difference(_ATOM_FEATURES)
            raise ValueError('Unrecognized atom features: {}'.format(', '.join(bad_features)))
        self.atom_features = atom_features
        if any(i not in _BOND_FEATURES for i in bond_features):
            bad_features = set(bond_features).difference(_BOND_FEATURES)
            raise ValueError('Unrecognized bond features: {}'.format(', '.join(bad_features)))
        self.bond_features = bond_features
        self.known_elements = known_elements
        self.distance_converter = distance_converter

    def convert(self, mol, state_attributes=None, full_pair_matrix=True):
        """
        Compute the representation for a molecule

        Args：
            mol (pybel.Molecule): Molecule to generate features for
            state_attributes (list): State attributes. Uses average mass and number of bonds per atom as default
            full_pair_matrix (bool): Whether to generate info for all atom pairs, not just bonded ones
        Returns:
            (dict): Dictionary of features
        """

        # Get the features features for all atoms and bonds
        atom_features = []
        atom_pairs = []
        for idx, atom in enumerate(mol.atoms):
            f = self.get_atom_feature(mol, atom)
            atom_features.append(f)
        atom_features = sorted(atom_features, key=lambda x: x["coordid"])
        num_atoms = mol.OBMol.NumAtoms()
        for i, j in itertools.combinations(range(0, num_atoms), 2):
            bond_feature = self.get_pair_feature(mol, i, j, full_pair_matrix)
            if bond_feature:
                atom_pairs.append(bond_feature)
            else:
                continue

        # Compute the graph distance, if desired
        if 'graph_distance' in self.bond_features:
            graph_dist = self._dijkstra_distance(atom_pairs)
            for i in atom_pairs:
                i.update({'graph_distance': graph_dist[i['a_idx'], i['b_idx']]})

        # Generate the state attributes (that describe the whole network)
        state_attributes = state_attributes or [
            [mol.molwt / num_atoms,
             len([i for i in atom_pairs if i['bond_type'] > 0]) / num_atoms]
        ]

        # Get the atom features in the order they are requested by the user as a 2D array
        atoms = []
        for atom in atom_features:
            atoms.append(self._create_atom_feature_vector(atom))

        # Get the bond features in the order request by the user
        bonds = []
        index1_temp = []
        index2_temp = []
        for bond in atom_pairs:
            # Store the index of each bond
            index1_temp.append(bond.pop('a_idx'))
            index2_temp.append(bond.pop('b_idx'))

            # Get the desired bond features
            bonds.append(self._create_pair_feature_vector(bond))

        # Given the bonds (i,j), make it so (i,j) == (j, i)
        index1 = index1_temp + index2_temp
        index2 = index2_temp + index1_temp
        bonds = bonds + bonds

        # Sort the arrays by the beginning index
        sorted_arg = np.argsort(index1)
        index1 = np.array(index1)[sorted_arg].tolist()
        index2 = np.array(index2)[sorted_arg].tolist()
        bonds = np.array(bonds)[sorted_arg].tolist()

        return {'atom': atoms,
                'bond': bonds,
                'state': state_attributes,
                'index1': index1,
                'index2': index2}

    def _create_pair_feature_vector(self, bond: dict) -> List[float]:
        """Generate the feature vector from the bond feature dictionary

        Handles the binarization of categorical variables, and performing the distance conversion

        Args:
            bond (dict): Features for a certain pair of atoms
        Returns:
            ([float]) Values converted to a vector
            """
        bond_temp = []
        for i in self.bond_features:
            # Some features require conversion (e.g., binarization)
            if i in bond:
                if i == "bond_type":
                    bond_temp.extend(fast_label_binarize(bond[i], [0, 1, 2, 3, 4]))
                elif i == "same_ring":
                    bond_temp.append(int(bond[i]))
                elif i == "spatial_distance":
                    expanded = self.distance_converter.convert([bond[i]])[0]
                    if isinstance(expanded, np.ndarray):
                        # If we use a distance expansion
                        bond_temp.extend(expanded.tolist())
                    else:
                        # If not
                        bond_temp.append(expanded)
                else:
                    bond_temp.append(bond[i])
        return bond_temp

    def _create_atom_feature_vector(self, atom: dict) -> List[int]:
        """Generate the feature vector from the atomic feature dictionary

        Handles the binarization of categorical variables, and transforming the ring_sizes to a list

        Args:
            atom (dict): Dictionary of atomic features
        Returns:
            ([int]): Atomic feature vector
        """
        atom_temp = []
        for i in self.atom_features:
            if i == 'chirality':
                atom_temp.extend(fast_label_binarize(atom[i], [0, 1, 2]))
            elif i == 'element':
                atom_temp.extend(fast_label_binarize(atom[i], self.known_elements))
            elif i in ['aromatic', 'donor', 'acceptor']:
                atom_temp.append(int(atom[i]))
            elif i == 'hybridization':
                atom_temp.extend(fast_label_binarize(atom[i], [1, 2, 3, 4, 5, 6]))
            elif i == 'ring_sizes':
                atom_temp.extend(ring_to_vector(atom[i]))
            else:  # It is a scalar
                atom_temp.append(atom[i])
        return atom_temp

    def _dijkstra_distance(self, pairs):
        """
        Compute the graph distance between each pair of atoms,
        using the network defined by the bonded atoms.

        Args:
            pairs ([dict]): List of bond information
        Returns:
            ([int]) Distance for each pair of bonds
        """
        bonds = []
        for p in pairs:
            if p['bond_type'] > 0:
                bonds.append([p['a_idx'], p['b_idx']])
        return dijkstra_distance(bonds)

    def get_atom_feature(self, mol, atom):
        """
        Generate all features of a particular atom

        Args:
            mol (pybel.Molecule): Molecule being evaluated
            atom (pybel.Atom): Specific atom being evaluated
        Return:
            (dict): All features for that atom
        """

        # Get the link to the OpenBabel representation of the atom
        obatom = atom.OBAtom
        atom_idx = atom.idx - 1  # (pybel atoms indices start from 1)

        # Get the element
        element = Element.from_Z(obatom.GetAtomicNum()).symbol

        # Get the fast-to-compute properties
        output = {"element": element,
                  "atomic_num": obatom.GetAtomicNum(),
                  "formal_charge": obatom.GetFormalCharge(),
                  "hybridization": 6 if element == 'H' else obatom.GetHyb(),
                  "acceptor": obatom.IsHbondAcceptor(),
                  "donor": obatom.IsHbondDonorH() if atom.type == 'H' else obatom.IsHbondDonor(),
                  "aromatic": obatom.IsAromatic(),
                  "coordid": atom.coordidx}

        # Get the chirality, if desired
        if 'chirality' in self.atom_features:
            # Determine whether the molecule has chiral centers
            chiral_cc = self._get_chiral_centers(mol)
            if atom_idx not in chiral_cc:
                output['chirality'] = 0
            else:
                # 1 --> 'R', 2 --> 'S'
                output['chirality'] = 1 if chiral_cc[atom_idx] == 'R' else 2

        # Find the rings, if desired
        if 'ring_sizes' in self.atom_features:
            rings = mol.OBMol.GetSSSR()  # OpenBabel caches ring computation internally, no need to cache ourselves
            output['ring_sizes'] = [r.Size() for r in rings if r.IsInRing(atom.idx)]

        return output

    def create_bond_feature(self, mol, bid, eid):
        """
        Create information for a bond for a pair of atoms that are not actually bonded

        Args:
            mol (pybel.Molecule): Molecule being featurized
            bid (int): Index of atom beginning of the bond
            eid (int): Index of atom at the end of the bond
        """
        a1 = mol.OBMol.GetAtom(bid + 1)
        a2 = mol.OBMol.GetAtom(eid + 1)
        same_ring = mol.OBMol.AreInSameRing(a1, a2)
        return {"a_idx": bid,
                "b_idx": eid,
                "bond_type": 0,
                "same_ring": True if same_ring else False,
                "spatial_distance": a1.GetDistance(a2)}

    def get_pair_feature(self, mol, bid, eid, full_pair_matrix):
        """
        Get the features for a certain bond

        Args:
            mol (pybel.Molecule): Molecule being featurized
            bid (int): Index of atom beginning of the bond
            eid (int): Index of atom at the end of the bond
            full_pair_matrix (bool): Whether to compute the matrix for every atom - even those that
                are not actually bonded
        """
        # Find the bonded pair of atoms
        bond = mol.OBMol.GetBond(bid + 1, eid + 1)
        if not bond:  # If the bond is ordered in the other direction
            bond = mol.OBMol.GetBond(eid + 1, bid + 1)

        # If the atoms are not bonded
        if not bond:
            if full_pair_matrix:
                return self.create_bond_feature(mol, bid, eid)
            else:
                return None

        # Compute bond features
        a1 = mol.OBMol.GetAtom(bid + 1)
        a2 = mol.OBMol.GetAtom(eid + 1)
        same_ring = mol.OBMol.AreInSameRing(a1, a2)
        return {"a_idx": bid,
                "b_idx": eid,
                "bond_type": 4 if bond.IsAromatic() else bond.GetBondOrder(),
                "same_ring": True if same_ring else False,
                "spatial_distance": a1.GetDistance(a2)}

    def _get_rdk_mol(self, mol, format='smiles'):
        """
        Return: RDKit Mol (w/o H)
        """
        if format == 'pdb':
            return Chem.rdmolfiles.MolFromPDBBlock(mol.write("pdb"))
        elif format == 'smiles':
            return Chem.rdmolfiles.MolFromSmiles(mol.write("smiles"))

    def _get_chiral_centers(self, mol):
        """
        Use RDKit to find the chiral centers with CIP(R/S) label

        This provides the absolute stereochemistry.  The chiral label obtained
        from pybabel and rdkit.mol.getchiraltag is relative positions of the bonds as provided

        Args:
            mol (Molecule): Molecule to asses
        Return:
            (dict): Keys are the atom index and values are the CIP label
        """
        mol_rdk = self._get_rdk_mol(mol, 'smiles')
        if mol_rdk is None:
            # Conversion to RDKit has failed
            return {}
        else:
            chiral_cc = Chem.FindMolChiralCenters(mol_rdk)
            return dict(chiral_cc)


def dijkstra_distance(bonds):
    """
    Compute the graph distance based on the dijkstra algorithm

    Args:
        bonds: (list of list), for example [[0, 1], [1, 2]] means two bonds formed by atom 0, 1 and atom 1, 2

    Returns:
        full graph distance matrix
    """
    nb_atom = max(itertools.chain(*bonds)) + 1
    graph_dist = np.ones((nb_atom, nb_atom), dtype=np.int32) * np.infty
    for bond in bonds:
        graph_dist[bond[0], bond[1]] = 1
        graph_dist[bond[1], bond[0]] = 1

    queue = deque()  # Queue used in all loops
    visited = set()  # Used in all loops
    for i in range(nb_atom):
        graph_dist[i, i] = 0
        visited.clear()
        queue.append(i)
        while queue:
            s = queue.pop()
            visited.add(s)

            for k in np.where(graph_dist[s, :] == 1)[0]:
                if k not in visited:
                    queue.append(k)
                    graph_dist[i, k] = min(graph_dist[i, k],
                                           graph_dist[i, s] + 1)
                    graph_dist[k, i] = graph_dist[i, k]
    return graph_dist


def mol_from_smiles(smiles):
    mol = pybel.readstring(format='smi', string=smiles)
    mol.make3D()
    return mol


def mol_from_pymatgen(mol):
    """
    Args:
        mol(Molecule)
    """
    mol = pybel.Molecule(BabelMolAdaptor(mol).openbabel_mol)
    mol.make3D()
    return mol


def mol_from_file(file_path, file_format='xyz'):
    """
    Args:
        file_path(str)
        file_format(str): allow formats that open babel supports
    """
    mol = [r for r in pybel.readfile(format=file_format,
                                     filename=file_path)][0]
    return mol


def _convert_mol(mol, molecule_format, converter):
    """Convert a molecule from string to its graph features

    Utility function used in the graph generator.

    The parse and convert operations are both in this function due to Pybel objects
    not being serializable. By not using the Pybel representation of each molecule
    as an input to this function, we can use multiprocessing to parallelize conversion
    over molecules as strings can be passed as pickle objects to the worker threads but
    but Pybel objects cannot.

    Args:
        mol (str): String representation of a molecule
        molecule_format (str): Format of the string representation
        converter (MolecularGraph): Tool used to generate graph representation
    Returns:
        (dict): Graph representation of the molecule
    """

    # Convert molecule into pybel format
    if molecule_format == 'smiles':
        mol = mol_from_smiles(mol)  # Used to generate 3D coordinates/H atoms
    else:
        mol = pybel.readstring(molecule_format, mol)

    return converter.convert(mol)


class MolecularGraphBatchGenerator(BaseGraphBatchGenerator):
    """Generator that creates batches of molecular data by computing graph properties on demand

    If your dataset is small enough that the descriptions of the whole dataset fit in memory,
    we recommend using :class:`megnet.data.graph.GraphBatchGenerator` instead to avoid
    the computational cost of dynamically computing graphs."""

    def __init__(self, mols, targets=None, converter=None, molecule_format='xyz',
                 batch_size=128, shuffle=True, n_jobs=1):
        """
        Args:
            mols ([str]): List of the string reprensetations of each molecule
            targets ([ndarray]): Properties of each molecule to be predicted
            converter (MolecularGraph): Converter used to generate graph features
            molecule_format (str): Format of each of the string representations in `mols`
            batch_size (int): Target size for each batch
            shuffle (bool): Whether to shuffle the training data after each epoch
            n_jobs (int): Number of worker threads (None to use all threads).
        """

        super().__init__(len(mols), targets, batch_size, shuffle)
        self.mols = np.array(mols)
        if converter is None:
            converter = MolecularGraph()
        self.converter = converter
        self.molecule_format = molecule_format
        self.n_jobs = n_jobs

        def mute():
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
        self.pool = Pool(self.n_jobs, initializer=mute) if self.n_jobs != 1 else None

    def __del__(self):
        if self.pool is not None:
            self.pool.close()  # Kill thread pool if generator is deleted

    def _generate_inputs(self, batch_index):
        # Get the molecules for this batch
        mols = self.mols[batch_index]

        # Generate the graphs
        graphs = self._generate_graphs(mols)

        # Return them as flattened into array format
        return self.converter.get_flat_data(graphs)

    def _generate_graphs(self, mols):
        """Generate graphs for a certain collection of molecules

        Args:
            mols ([string]): Molecules to process
        Returns:
            ([dict]): Graphs for all of the molecules
        """
        if self.pool is None:
            graphs = [_convert_mol(m, self.molecule_format, self.converter) for m in mols]
        else:
            func = partial(_convert_mol, molecule_format=self.molecule_format,
                           converter=self.converter)
            graphs = self.pool.map(func, mols)
        return graphs

    def create_cached_generator(self) -> GraphBatchGenerator:
        """Generates features for all of the molecules and stores them in memory

        Returns:
            (GraphBatchGenerator) Graph genereator that relies on having the graphs in memory
        """

        # Make all the graphs
        graphs = self._generate_graphs(self.mols)

        # Turn them into a fat array
        inputs = self.converter.get_flat_data(graphs, self.targets)

        return GraphBatchGenerator(*inputs, is_shuffle=self.is_shuffle,
                                   batch_size=self.batch_size)
