from megnet.models import MEGNetModel
from monty.serialization import loadfn
import os
from pymatgen.io.babel import BabelMolAdaptor
import logging

try:
    import pybel as pb
except:
    logging.warning("Openbabel is needed for molecule models, try 'conda install -c openbabel openbabel' to install it")
    pb = None


ATOMNUM2TYPE = {1: 1, 6: 2, 7: 4, 8: 6, 9: 8}
pjoin = os.path.join
QM9_MODELDIR = pjoin(os.path.dirname(__file__), '../../mvl_models/qm9-2018.6.1')
SCALER = loadfn(pjoin(QM9_MODELDIR, "scaler.json"))


def get_pmg_mol_from_smiles(smiles):
    """
    Get a pymatgen molecule from smiles representation
    Args:
        smiles: (str) smiles representation of molecule
    """
    b_mol = pb.readstring('smi', smiles)
    b_mol.make3D()
    b_mol = b_mol.OBMol
    p_mol = BabelMolAdaptor(b_mol).pymatgen_mol
    return p_mol


class AtomNumberToTypeConvertor:
    """
    A convertor that takes atomic number list and map
    it to type list as in qm9 dataset
    """
    def __init__(self, mapping=ATOMNUM2TYPE):
        self.mapping = mapping

    def convert(self, l):
        return [self.mapping[i] for i in l]

class Scaler:
    """
    simiple standard scaler with option to 
    consider per atom quantities. If the predicted 
    properties is a per atom quantity the final 
    result will multiply by the number of atoms
    """
    def __init__(self, mean, std, is_pa):
        self.mean = mean
        self.std = std
        self.is_pa = is_pa 

    def transform(self, target, structure):
        if self.is_pa:
            n = len(structure)
        else:
            n = 1
        return n * (target * self.std + self.mean)


class QM9Model:
    """
    Wrapper around our pretrained QM9 dataset

    Args:
        target_name: (str) QM9 property name

    Methods:
        predict_structure(structure): compute the model prediction for structure
        predict_smiles(smiles): compute the model prediction for smiles representation of molecules
    """
    def __init__(self, target_name):
        self.model = MEGNetModel.from_file(pjoin(QM9_MODELDIR, target_name+".hdf5"))
        self.model.graph_convertor.atom_convertor = AtomNumberToTypeConvertor() 
        self.scaler = Scaler(SCALER[target_name]['mean'], SCALER[target_name]['std'], SCALER[target_name]['is_per_atom'])

    def predict_structure(self, structure):
        """
        Predict the property of structure

        Args:
            structure: (pymatgen molecule)

        Returns:
            target: (float)
        """
        target = self.scaler.transform(
                self.model.predict_structure(structure), 
                structure
                )
        return target[0]

    def predict_smiles(self, smiles):
        """
        Predict the property of smiles

        Args:
            smiles: (str) smiles representation

        Returns:
            target: (float)
        """
        mol = get_pmg_mol_from_smiles(smiles)
        return self.predict_structure(mol)
