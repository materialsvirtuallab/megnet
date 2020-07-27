"""
This module implements atom/bond/structure-wise descriptor calculated from
pretrained megnet model
"""

import os
from typing import Union, Dict

import numpy as np
from tensorflow.keras.models import Model

from megnet.models import MEGNetModel, GraphModel
from megnet.utils.typing import StructureOrMolecule

DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__),
    '../../mvl_models/mp-2019.4.1/formation_energy.hdf5')


class MEGNetDescriptor:
    """
    MEGNet descriptors. This class takes a trained model and
    then compute the intermediate outputs as structure features

    """
    def __init__(self,
                 model_name: Union[str, GraphModel,
                                   MEGNetModel] = DEFAULT_MODEL,
                 use_cache: bool = True):
        """
        Args:
            model_name (str or MEGNetModel): trained model. If it is
                str, then only models in mvl_models are used.
            use_cache (bool): whether to use cache for structure
                graph calculations
        """
        if isinstance(model_name, str):
            model = MEGNetModel.from_file(model_name)
        elif isinstance(model_name, GraphModel):
            model = model_name
        else:
            raise ValueError('model_name only support str '
                             'or GraphModel object')

        layers = model.layers
        important_prefix = ['meg', 'set', 'concatenate']

        all_names = [i.name for i in layers
                     if any([i.name.startswith(j) for j in important_prefix])]

        if any([i.startswith('megnet') for i in all_names]):
            self.version = 'v2'
        else:
            self.version = 'v1'

        valid_outputs = [i.output for i in layers
                         if any([i.name.startswith(j) for j in important_prefix])]

        outputs = []
        valid_names = []
        for i, j in zip(all_names, valid_outputs):
            if isinstance(j, list):
                for k, l in enumerate(j):
                    valid_names.append(i + '_%d' % k)
                    outputs.append(l)
            else:
                valid_names.append(i)
                outputs.append(j)

        full_model = Model(inputs=model.inputs, outputs=outputs)
        model.model = full_model
        self.model = model
        self.valid_names = valid_names
        self._cache: Dict[str, float] = {}
        self.use_cache = use_cache

    def _predict_structure(self, structure: StructureOrMolecule) -> np.ndarray:
        graph = self.model.graph_converter.convert(structure)
        inp = self.model.graph_converter.graph_to_input(graph)
        return self.model.predict(inp)

    def _predict_feature(self, structure: StructureOrMolecule) -> np.ndarray:
        if not self.use_cache:
            return self._predict_structure(structure)

        s = str(structure)
        if s in self._cache:
            return self._cache[s]
        result = self._predict_structure(structure)
        self._cache[s] = result
        return result

    def _get_features(self,
                      structure: StructureOrMolecule,
                      prefix: str,
                      level: int,
                      index: int = None) -> np.ndarray:
        name = prefix
        if level is not None:
            name = prefix + "_%d" % level
        if index is not None:
            name += '_%d' % index

        if name not in self.valid_names:
            raise ValueError("%s not in original megnet model" % name)
        ind = self.valid_names.index(name)
        out_all = self._predict_feature(structure)
        return out_all[ind][0]

    def _get_updated_prefix_level(self, prefix: str, level: int):
        mapping = {'meg_net_layer': ["megnet", level-1],
                   "set2_set": ["set2set_atom" if level == 1 else "set2set_bond", None],
                   "concatenate": ["concatenate", None]}
        if self.version == "v2":
            return mapping[prefix][0], mapping[prefix][1]  # type: ignore
        return prefix, level

    def get_atom_features(self, structure: StructureOrMolecule,
                          level: int = 3) -> np.ndarray:
        """
        Get megnet atom features from structure
        Args:
            structure: pymatgen structure or molecule
            level: int, indicating the block number of megnet, starting
                from 1

        Returns:
            nxm atomic feature matrix

        """
        prefix, level = self._get_updated_prefix_level('meg_net_layer', level)
        return self._get_features(structure, prefix=prefix,
                                  level=level, index=0)

    def get_bond_features(self, structure: StructureOrMolecule,
                          level: int = 3) -> np.ndarray:
        """
        Get bond features at megnet block level
        Args:
            structure: pymatgen structure
            level: int

        Returns:
            n_bond x m bond feature matrix

        """
        prefix, level = self._get_updated_prefix_level('meg_net_layer', level)
        return self._get_features(structure, prefix=prefix,
                                  level=level, index=1)

    def get_global_features(self, structure: StructureOrMolecule,
                            level: int = 2) -> np.ndarray:
        """
        Get state features at megnet block level
        Args:
            structure: pymatgen structure or molecule
            level: int

        Returns:
            1 x m_g global feature vector

        """
        prefix, level = self._get_updated_prefix_level('meg_net_layer', level)
        return self._get_features(structure,
                                  prefix=prefix, level=level, index=2)

    def get_set2set(self, structure: StructureOrMolecule,
                    ftype: str = 'atom') -> np.ndarray:
        """
        Get set2set output as features
        Args:
            structure (StructureOrMolecule): pymatgen structure
                or molecule
            ftype (str): atom or bond

        Returns:
            feature matrix, each row is a vector for an atom
            or bond

        """
        mapping = {'atom': 1, 'bond': 2}
        prefix, level = self._get_updated_prefix_level('set2_set', level=mapping[ftype])
        return self._get_features(structure, prefix=prefix, level=level)

    def get_structure_features(self, structure: StructureOrMolecule) -> np.ndarray:
        """
        Get structure level feature vector
        Args:
            structure (StructureOrMolecule): pymatgen structure
                or molecule

        Returns:
            one feature vector for the structure

        """
        prefix, level = self._get_updated_prefix_level('concatenate', level=1)
        return self._get_features(structure, prefix=prefix, level=level)
