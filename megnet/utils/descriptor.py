"""
This module implements atom/bond/structure-wise descriptor calculated from
pretrained megnet model
"""

import os
from megnet.models import MEGNetModel, GraphModel
from keras.models import Model

DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), '../../mvl_models/mp-2019.4.1/formation_energy.hdf5')


class MEGNetDescriptor:
    def __init__(self, model_name=DEFAULT_MODEL, use_cache=True):
        if isinstance(model_name, str):
            model = MEGNetModel.from_file(model_name)
        elif isinstance(model_name, GraphModel):
            model = model_name
        else:
            raise ValueError('model_name only support str or GraphModel object')

        layers = model.layers
        important_prefix = ['meg', 'set', 'concatenate']
        all_names = [i.name for i in layers if any([i.name.startswith(j) for j in important_prefix])]
        valid_outputs = [i.output for i in layers if any([i.name.startswith(j) for j in important_prefix])]

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
        self._cache = {}
        self.use_cache = use_cache

    def _predict_structure(self, structure):
        graph = self.model.graph_converter.convert(structure)
        inp = self.model.graph_converter.graph_to_input(graph)
        return self.model.predict(inp)

    def _predict_feature(self, structure):
        if not self.use_cache:
            return self._predict_structure(structure)

        s = str(structure)
        if s in self._cache:
            return self._cache[s]
        else:
            result = self._predict_structure(structure)
            self._cache[s] = result
            return result

    def _get_features(self, structure, prefix, level, index=None):
        name = prefix + "_%d" % level
        if index is not None:
            name += '_%d' % index

        if name not in self.valid_names:
            raise ValueError("%s not in original megnet model" % name)
        ind = self.valid_names.index(name)
        out_all = self._predict_feature(structure)
        return out_all[ind][0]

    def get_atom_features(self, structure, level=2):
        """
        Get megnet atom features from structure
        Args:
            structure: pymatgen structure
            level: int, indicating the block number of megnet, starting from 1

        Returns:
            nxm atomic feature matrix

        """
        return self._get_features(structure, prefix='meg_net_layer', level=level, index=0)

    def get_bond_features(self, structure, level=2):
        """
        Get bond features at megnet block level
        Args:
            structure: pymatgen structure
            level: int

        Returns:
            n_bond x m bond feature matrix

        """
        return self._get_features(structure, prefix='meg_net_layer', level=level, index=1)

    def get_global_features(self, structure, level=2):
        """
        Get state features at megnet block level
        Args:
            structure: pymatgen structure
            level: int

        Returns:
            1 x m_g global feature vector

        """
        return self._get_features(structure, prefix='meg_net_layer', level=level, index=2)

    def get_set2set(self, structure, ftype='atom'):
        mapping = {'atom': 1, 'bond': 2}
        return self._get_features(structure, prefix='set2_set', level=mapping[ftype])

    def get_structure_features(self, structure):
        return self._get_features(structure, prefix='concatenate', level=1)
