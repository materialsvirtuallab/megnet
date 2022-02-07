"""
Molecule utility, mainly using openbabel
"""
import logging

from monty.dev import requires
from pymatgen.core import Molecule

try:
    from openbabel import openbabel as ob  # type: ignore
    from openbabel import pybel as pb  # type: ignore
except ImportError:
    logging.warning(
        "Openbabel is needed for molecule models, " "try 'conda install -c openbabel openbabel' " "to install it"
    )
    pb = None
    ob = None


@requires(pb is not None, "openbabel is needed to run convert smiles")
def get_pmg_mol_from_smiles(smiles: str) -> Molecule:
    """
    Get a pymatgen molecule from smiles representation
    Args:
        smiles: (str) smiles representation of molecule

    Returns:
        pymatgen Molecule
    """
    b_mol = pb.readstring("smi", smiles)  # noqa
    b_mol.make3D()
    b_mol = b_mol.OBMol
    sp = []
    coords = []
    for atom in ob.OBMolAtomIter(b_mol):
        sp.append(atom.GetAtomicNum())
        coords.append([atom.GetX(), atom.GetY(), atom.GetZ()])
    return Molecule(sp, coords)
