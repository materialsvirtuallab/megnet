"""
Tools for creating graph inputs from molecule data
"""

import re
import itertools


from typing import List
from pymatgen import Molecule, Element
from pymatgen.io.babel import BabelMolAdaptor
import numpy as np
from megnet.data.qm9 import ring_to_vector
from megnet.data.graph import StructureGraph, GaussianDistance
from sklearn.preprocessing import label_binarize

try:
    import pybel
except:
    pybel = None

try:
    from rdkit import Chem
except:
    Chem = None

__date__ = '12/01/2018'

# List of all possible atomic features
ATOM_FEATURES = ['atomic_num', 'chirality', 'formal_charge', 'ring_sizes',
                 'hybridization', 'donor', 'acceptor', 'aromatic']

# List of all possible bond features
BOND_FEATURES = ['bond_type', 'same_ring', 'spatial_distance', 'graph_distance']


class SimpleMolGraph(StructureGraph):
    """
    Default using all atom pairs as bonds. The distance between atoms are used
    as bond features. By default the distance is expanded using a Gaussian
    expansion with centers at np.linspace(0, 4, 20) and width of 0.5
    """
    def __init__(self,
                 nn_strategy='AllAtomPairs',
                 atom_convertor=None,
                 bond_convertor=None
                 ):
        if bond_convertor is None:
            bond_convertor = GaussianDistance(np.linspace(0, 4, 20), 0.5)
        super().__init__(nn_strategy=nn_strategy, atom_convertor=atom_convertor,
                         bond_convertor=bond_convertor)


class MolecularGraph(StructureGraph):
    """Class for generating the graph inputs from a molecule

    Computes many different features for the atoms and bonds in a molecule, and prepares them
    in a form compatible with MEGNet models. The :meth:`convert` method takes a OpenBabel molecule
    and, besides computing features, also encodes them in a form compatible with machine learning.
    Namely, the `convert` method one-hot encodes categorical variables and concatenates the atomic features

    ## Atomic Features

    This class can compute the following features for each atom

    - `atomic_num`: The atomic number
    - `chirality`: (categorical) R, S, or not a Chiral center (one-hot encoded).
    - `formal_charge`: Formal charge of the atom
    - `ring_sizes`: For rings with 9 or fewer atoms, how many unique rings of each size include this atom
    - `hybridization`: (categorical) Hybridization of atom: sp, sp2, sp3, sq. planer, trig, octahedral, or hydrogen
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

    The class may use the distance

    """
    def __init__(self, atom_features=None, bond_features=None, distance_converter=None):
        """
        Args:
            atom_features ([str]): List of atom features to compute
            bond_features ([str]): List of bond features to compute
        """
        # TODO (wardlt): I do not think NN strategy is not actually used by this class. Refactor StructureGraph?
        super().__init__('AllAtomPairs')
        if bond_features is None:
            bond_features = BOND_FEATURES
        if atom_features is None:
            atom_features = ATOM_FEATURES
        if distance_converter is None:
            distance_converter = GaussianDistance(np.linspace(0, 4, 20), 0.5)

        # Check if all feature names are valid
        if any(i not in ATOM_FEATURES for i in atom_features):
            bad_features = set(atom_features).difference(ATOM_FEATURES)
            raise ValueError('Unrecognized atom features: {}'.format(', '.join(bad_features)))
        self.atom_features = atom_features
        if any(i not in BOND_FEATURES for i in bond_features):
            bad_features = set(bond_features).difference(BOND_FEATURES)
            raise ValueError('Unrecognized bond features: {}'.format(', '.join(bad_features)))
        self.bond_features = bond_features
        self.distance_converter = distance_converter

    def convert(self, mol: pybel.Molecule, state_attributes=None, full_pair_matrix=True):
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
                    bond_temp.extend(label_binarize([bond[i]], range(5))[0].tolist())
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
            # TODO (wardlt): One-hot encoding for the elements
            if i == 'chirality':
                atom_temp.extend(label_binarize([atom[i]], [0, 1, 2])[0].tolist())
            elif i in ['aromatic', 'donor', 'acceptor']:
                atom_temp.extend(label_binarize([atom[i]], [False, True])[0].tolist())
            elif i == 'hybridization':
                atom_temp.extend(label_binarize([atom[i]], range(1, 7))[0].tolist())
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
        a1 = mol.atoms[bid].OBAtom
        a2 = mol.atoms[eid].OBAtom
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
        a1 = mol.atoms[bid].OBAtom
        a2 = mol.atoms[eid].OBAtom
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
        chiral_cc = Chem.FindMolChiralCenters(mol_rdk)
        return dict(chiral_cc)


def dijkstra_distance(bonds):
    """
    Compute the graph distance based on the dijkstra algorithm
    :param bonds: (list of list), for example [[0, 1], [1, 2]] means two bonds formed by atom 0, 1 and atom 1, 2
    :return: full graph distance matrix
    """
    nb_atom = max(itertools.chain(*bonds)) + 1
    graph_dist = np.ones((nb_atom, nb_atom), dtype=np.int32) * np.infty
    for bond in bonds:
        graph_dist[bond[0], bond[1]] = 1
        graph_dist[bond[1], bond[0]] = 1

    for i in range(nb_atom):
        graph_dist[i, i] = 0
        unvisited = list(range(nb_atom))
        visited = []
        queue = []
        unvisited.remove(i)
        queue.append(i)
        while queue:
            s = queue.pop(0)
            visited.append(s)

            for k in np.where(graph_dist[s, :] == 1)[0]:
                if k not in visited:
                    queue.append(k)
                    # print(s, k, visited)
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
