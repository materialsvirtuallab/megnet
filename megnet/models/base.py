"""
Implements basic GraphModels.
"""

import os
from typing import Dict, List, Union
from warnings import warn

import numpy as np
from monty.serialization import dumpfn, loadfn
from tensorflow.keras.backend import int_shape
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from pymatgen import Structure

from megnet.callbacks import ModelCheckpointMAE, ManualStop, ReduceLRUponNan
from megnet.data.graph import GraphBatchDistanceConvert, GraphBatchGenerator, \
    StructureGraph
from megnet.utils.preprocessing import DummyScaler, Scaler


class GraphModel:
    """
    Composition of keras model and converter class for transfering structure
    object to input tensors. We add methods to train the model from
    (structures, targets) pairs
    """

    def __init__(self,
                 model: Model,
                 graph_converter: StructureGraph,
                 target_scaler: Scaler = DummyScaler(),
                 metadata: Dict = None,
                 **kwargs):
        """
        Args:
            model: (keras model)
            graph_converter: (object) a object that turns a structure to a graph,
                check `megnet.data.crystal`
            target_scaler: (object) a scaler object for converting targets, check
                `megnet.utils.preprocessing`
            metadata: (dict) An optional dict of metadata associated with the model.
                Recommended to incorporate some basic information such as units,
                MAE performance, etc.
        """
        self.model = model
        self.graph_converter = graph_converter
        self.target_scaler = target_scaler
        self.metadata = metadata or {}

    def __getattr__(self, p):
        return getattr(self.model, p)

    def train(self,
              train_structures: List[Structure],
              train_targets: List[float],
              validation_structures: List[Structure] = None,
              validation_targets: List[float] = None,
              sample_weights: List[float] = None,
              epochs: int = 1000,
              batch_size: int = 128,
              verbose: int = 1,
              callbacks: List[Callback] = None,
              scrub_failed_structures: bool = False,
              prev_model: str = None,
              save_checkpoint: bool = True,
              automatic_correction: bool = True,
              lr_scaling_factor: float = 0.5,
              patience: int = 500,
              **kwargs) -> "GraphModel":
        """
        Args:
            train_structures: (list) list of pymatgen structures
            train_targets: (list) list of target values
            validation_structures: (list) list of pymatgen structures as validation
            validation_targets: (list) list of validation targets
            sample_weights: (list) list of sample weights for training data
            epochs: (int) number of epochs
            batch_size: (int) training batch size
            verbose: (int) keras fit verbose, 0 no progress bar, 1 only at the epoch end and 2 every batch
            callbacks: (list) megnet or keras callback functions for training
            scrub_failed_structures: (bool) whether to scrub structures with failed graph computation
            prev_model: (str) file name for previously saved model
            save_checkpoint: (bool) whether to save checkpoint
            automatic_correction: (bool) correct nan errors
            lr_scaling_factor: (float, less than 1) scale the learning rate down when nan loss encountered
            patience: (int) patience for early stopping
            **kwargs:
        """
        train_graphs, train_targets = self.get_all_graphs_targets(train_structures, train_targets,
                                                                  scrub_failed_structures=scrub_failed_structures)
        if (validation_structures is not None) and (validation_targets is not None):
            val_graphs, validation_targets = self.get_all_graphs_targets(
                validation_structures, validation_targets, scrub_failed_structures=scrub_failed_structures)
        else:
            val_graphs = None

        self.train_from_graphs(train_graphs,
                               train_targets,
                               validation_graphs=val_graphs,
                               validation_targets=validation_targets,
                               sample_weights=sample_weights,
                               epochs=epochs,
                               batch_size=batch_size,
                               verbose=verbose,
                               callbacks=callbacks,
                               prev_model=prev_model,
                               lr_scaling_factor=lr_scaling_factor,
                               patience=patience,
                               save_checkpoint=save_checkpoint,
                               automatic_correction=automatic_correction,
                               **kwargs
                               )
        return self

    def train_from_graphs(self,
                          train_graphs: List[Dict],
                          train_targets: List[float],
                          validation_graphs: List[Dict] = None,
                          validation_targets: List[float] = None,
                          sample_weights: List[float] = None,
                          epochs: int = 1000,
                          batch_size: int = 128,
                          verbose: int = 1,
                          callbacks: List[Callback] = None,
                          prev_model: str = None,
                          lr_scaling_factor: float = 0.5,
                          patience: int = 500,
                          save_checkpoint: bool = True,
                          automatic_correction: bool = True,
                          **kwargs
                          ) -> "GraphModel":
        """
        Args:
            train_graphs: (list) list of graph dictionaries
            train_targets: (list) list of target values
            validation_graphs: (list) list of graphs as validation
            validation_targets: (list) list of validation targets
            sample_weights: (list) list of sample weights
            epochs: (int) number of epochs
            batch_size: (int) training batch size
            verbose: (int) keras fit verbose, 0 no progress bar, 1 only at the epoch end and 2 every batch
            callbacks: (list) megnet or keras callback functions for training
            prev_model: (str) file name for previously saved model
            lr_scaling_factor: (float, less than 1) scale the learning rate down when nan loss encountered
            patience: (int) patience for early stopping
            save_checkpoint: (bool) whether to save checkpoint
            automatic_correction: (bool) correct nan errors
            **kwargs:
        """
        # load from saved model
        if prev_model:
            self.load_weights(prev_model)
        is_classification = 'entropy' in str(self.model.loss)
        monitor = 'val_acc' if is_classification else 'val_mae'
        mode = 'max' if is_classification else 'min'
        dirname = kwargs.pop('dirname', 'callback')
        has_sample_weights = sample_weights is not None
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if callbacks is None:
            # with this call back you can stop the model training by `touch STOP`
            callbacks = [ManualStop()]
        train_nb_atoms = [len(i['atom']) for i in train_graphs]
        train_targets = [self.target_scaler.transform(i, j) for i, j in zip(train_targets, train_nb_atoms)]
        if (validation_graphs is not None) and (validation_targets is not None):
            filepath = os.path.join(dirname, '%s_{epoch:05d}_{%s:.6f}.hdf5' % (monitor, monitor))
            val_nb_atoms = [len(i['atom']) for i in validation_graphs]
            validation_targets = [self.target_scaler.transform(i, j) for i, j in zip(validation_targets, val_nb_atoms)]
            val_inputs = self.graph_converter.get_flat_data(validation_graphs, validation_targets)

            val_generator = self._create_generator(*val_inputs,
                                                   batch_size=batch_size)
            steps_per_val = int(np.ceil(len(validation_graphs) / batch_size))
            if save_checkpoint:
                callbacks.extend([ModelCheckpointMAE(filepath=filepath,
                                                     monitor=monitor,
                                                     mode=mode,
                                                     save_best_only=True,
                                                     save_weights_only=False,
                                                     val_gen=val_generator,
                                                     steps_per_val=steps_per_val,
                                                     target_scaler=self.target_scaler)])
                # avoid running validation twice in an epoch
                val_generator = None  # type: ignore
                steps_per_val = None  # type: ignore

            if automatic_correction:
                callbacks.extend([ReduceLRUponNan(filepath=filepath,
                                                  monitor=monitor,
                                                  mode=mode,
                                                  factor=lr_scaling_factor,
                                                  patience=patience,
                                                  has_sample_weights=has_sample_weights
                                                  )])
        else:
            val_generator = None  # type: ignore
            steps_per_val = None  # type: ignore

        train_inputs = self.graph_converter.get_flat_data(train_graphs, train_targets)
        # check dimension match
        self.check_dimension(train_graphs[0])
        train_generator = self._create_generator(*train_inputs, sample_weights=sample_weights,
                                                 batch_size=batch_size)
        steps_per_train = int(np.ceil(len(train_graphs) / batch_size))
        self.fit(train_generator, steps_per_epoch=steps_per_train,
                 validation_data=val_generator, validation_steps=steps_per_val,
                 epochs=epochs, verbose=verbose, callbacks=callbacks, **kwargs)
        return self

    def check_dimension(self, graph: Dict) -> bool:
        """
        Check the model dimension against the graph converter dimension
        Args:
            graph: structure graph

        Returns:

        """
        test_inp = self.graph_converter.graph_to_input(graph)
        input_shapes = [i.shape for i in test_inp]

        model_input_shapes = [int_shape(i) for i in self.model.inputs]

        def _check_match(real_shape, tensor_shape):
            if len(real_shape) != len(tensor_shape):
                return False
            matched = True
            for i, j in zip(real_shape, tensor_shape):
                if j is None:
                    continue

                if i == j:
                    continue
                matched = False
            return matched

        for i, j, k in zip(['atom features', 'bond features', 'state features'],
                           input_shapes[:3], model_input_shapes[:3]):
            matched = _check_match(j, k)
            if not matched:
                raise ValueError("The data dimension for %s is %s and does not match model "
                                 "required shape of %s" % (i, str(j), str(k)))
        return False

    def get_all_graphs_targets(self, structures: List[Structure],
                               targets: List[float],
                               scrub_failed_structures: bool = False) -> tuple:
        """
        Compute the graphs from structures and spit out (graphs, targets) with options to
        automatically remove structures with failed graph computations

        Args:
            structures: (list) pymatgen structure list
            targets: (list) target property list
            scrub_failed_structures: (bool) whether to scrub those failed structures

        Returns:
            graphs, targets

        """
        graphs_valid = []
        targets_valid = []

        for i, (s, t) in enumerate(zip(structures, targets)):
            try:
                graph = self.graph_converter.convert(s)
                graphs_valid.append(graph)
                targets_valid.append(t)
            except Exception as e:
                if scrub_failed_structures:
                    warn("structure with index %d failed the graph computations" % i,
                         UserWarning)
                    continue
                raise RuntimeError(str(e))
        return graphs_valid, targets_valid

    def predict_structure(self, structure: Structure) -> np.ndarray:
        """
        Predict property from structure

        Args:
            structure: pymatgen structure or molecule

        Returns:
            predicted target value
        """
        graph = self.graph_converter.convert(structure)
        return self.predict_graph(graph)

    def predict_graph(self, graph: Dict) -> np.ndarray:
        """
        Predict property from graph

        Args:
            graph: a graph dictionary, see megnet.data.graph

        Returns:
            predicted target value

        """
        inp = self.graph_converter.graph_to_input(graph)
        pred = self.predict(inp)  # direct prediction, shape [1, 1, m]
        return self.target_scaler.inverse_transform(pred[0, 0],
                                                    len(graph['atom']))

    def _create_generator(self, *args, **kwargs) -> \
            Union[GraphBatchDistanceConvert, GraphBatchGenerator]:
        if hasattr(self.graph_converter, 'bond_converter'):
            kwargs.update({'distance_converter': self.graph_converter.bond_converter})
            return GraphBatchDistanceConvert(*args, **kwargs)
        return GraphBatchGenerator(*args, **kwargs)

    def save_model(self, filename: str) -> None:
        """
        Save the model to a keras model hdf5 and a json config for additional
        converters

        Args:
            filename: (str) output file name

        Returns:
            None
        """
        self.model.save(filename)
        dumpfn(
            {
                'graph_converter': self.graph_converter,
                'target_scaler': self.target_scaler,
                'metadata': self.metadata
            },
            filename + '.json'
        )

    @classmethod
    def from_file(cls, filename: str) -> 'GraphModel':
        """
        Class method to load model from
            filename for keras model
            filename.json for additional converters

        Args:
            filename: (str) model file name

        Returns
            GraphModel
        """
        configs = loadfn(filename + '.json')
        from tensorflow.keras.models import load_model
        from megnet.layers import _CUSTOM_OBJECTS
        model = load_model(filename, custom_objects=_CUSTOM_OBJECTS)
        configs.update({'model': model})
        return GraphModel(**configs)
