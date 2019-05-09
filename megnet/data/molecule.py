"""
Create the graph from molecules. This implementation is in alpha version.
The original MEGNet paper uses features directly taken from Faber et al. at
https://drive.google.com/open?id=0Bzn36Iqm8hZscHFJcVh5aC1mZFU

"""
import itertools
import re
from pymatgen import Molecule
from pymatgen.io.babel import BabelMolAdaptor
import numpy as np
from megnet.data.graph import StructureGraph, GaussianDistance

try:
    import pybel
except:
    pybel = None

try:
    from rdkit import Chem
except:
    Chem = None

__date__ = '12/01/2018'

ATOM_FEATURES = ['atomic_num', 'chirality', 'partial_charge', 'ring_sizes',
                 'hybridization', 'donor', 'acceptor', 'aromatic']
BOND_FEATURES = ['a_idx', 'b_idx', 'bond_type', 'same_ring', 'spatial_distance',
                 'graph_distance']


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
    def __init__(self,
                 atom_features=ATOM_FEATURES,
                 bond_features=BOND_FEATURES):
        """
        Args:
            mol (pybel.Molecule)
        """
        self.atom_features = atom_features
        self.bond_features = bond_features

    def convert(self, mol, state_attributes=None, full_pair_matrix=True):
        """
        Args：
            mol: (object)
            state_attributes: (list) state attributes
            full_pair_matrix:
                Whether to get to full matrix instead of half
                Default true
        Returns:
            atom_features matrix, bond_features_matrix
        """
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
        graph_dist = self._dijkstra_distance(atom_pairs)
        for i in atom_pairs:
            i.update({'graph_distance': graph_dist[i['a_idx'], i['b_idx']]})

        out_atom = []
        out_pair = []
        for i in atom_features:
            d = dict()
            for j, k in i.items():
                if j in self.atom_features:
                    d.update({j: k})
            out_atom.append(d)

        for i in atom_pairs:
            d = dict()
            for j, k in i.items():
                if j in self.bond_features:
                    d.update({j: k})
            out_pair.append(d)

        state_attributes = state_attributes or [[0, 0]]
        atoms = []
        bonds = []
        index1_temp = []
        index2_temp = []
        for atom in out_atom:
            atom_temp = []
            for i in self.atom_features:
                if i in atom:
                    atom_temp.append(atom[i])
            atoms.append(atom_temp)

        for bond in out_pair:
            index1_temp.append(bond.pop('a_idx'))
            index2_temp.append(bond.pop('b_idx'))
            bond_temp = []
            for i in self.bond_features:
                if i in bond:
                    bond_temp.append(bond[i])
            bonds.append(bond_temp)

        index1 = index1_temp + index2_temp
        index2 = index2_temp + index1_temp
        bonds = bonds + bonds

        sorted_arg = np.argsort(index1)
        index1 = np.array(index1)[sorted_arg].tolist()
        index2 = np.array(index2)[sorted_arg].tolist()
        bonds = np.array(bonds)[sorted_arg].tolist()

        return {'atom': atoms,
                'bond': bonds,
                'state': state_attributes,
                'index1': index1,
                'index2': index2
                }

    def _dijkstra_distance(self, pairs):
        bonds = []
        for p in pairs:
            if p['bond_type'] > 0:
                bonds.append([p['a_idx'], p['b_idx']])
        return dijkstra_distance(bonds)

    def get_atom_feature(self, mol, atom):
        """
        Args:
            atom(pybel.Atom)
        Return:
        """
        obatom = atom.OBAtom
        atom_idx = atom.idx - 1  # (pybel atoms indexs start from 1)
        chiral_cc = self._get_chiral_centers(mol)
        if atom_idx not in chiral_cc:
            chirality = 0
        else:
            # 1 --> 'R', 2 --> 'S'
            chirality = 1 if chiral_cc[atom_idx] == 'R' else 2
        element = re.findall(r'\D+', atom.type)[0]
        return {"element": element,
                "atomic_num": obatom.GetAtomicNum(),
                "chirality": chirality,
                "formal_charge": obatom.GetFormalCharge(),
                "ring_sizes": [i for i in range(3, 9) if
                               obatom.IsInRingSize(i)],
                "hybridization": 6 if element == 'H' else obatom.GetHyb(),
                "acceptor": obatom.IsHbondAcceptor(),
                "donor": obatom.IsHbondDonorH() if atom.type == 'H' else obatom.IsHbondDonor(),
                "aromatic": obatom.IsAromatic(),
                "coordid": atom.coordidx}

    def create_bond_feature(self, mol, bid, eid):
        """
        Function to create the bond if there isn't the bond info with pybel
        Args:
            bid(int), eid(int): begin and end atoms' index
                                # Start from 0
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
        Args:
            bond(pybel.OBBond)
            mol(pybel.Molecule)
        """
        bond = mol.OBMol.GetBond(bid + 1, eid + 1)
        if not bond:
            bond = mol.OBMol.GetBond(eid + 1, bid + 1)
        if not bond:
            if full_pair_matrix:
                return self.create_bond_feature(mol, bid, eid)
            else:
                return None
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
        This provides the absolute stereochemistry
        The chiralabel obtaiend from pybabel and rdkit.mol.getchiraltag
        is relative positions of the bonds as provided
        Return: List of chiral centers with CIP label
        eg. [(1,'S'), (3, 'R')]
        """
        mol_rdk = self._get_rdk_mol(mol, 'smiles')
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
