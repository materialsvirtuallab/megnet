"""
Implements basic GraphModels.
"""
from __future__ import annotations

import os
from warnings import warn

import numpy as np
from monty.serialization import dumpfn, loadfn
from pymatgen.core import Structure
from tensorflow.keras.backend import int_shape
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tqdm import tqdm

from megnet.callbacks import EarlyStopping, ManualStop, ModelCheckpointMAE
from megnet.data.graph import (
    GraphBatchDistanceConvert,
    GraphBatchGenerator,
    StructureGraph,
)
from megnet.utils.preprocessing import DummyScaler, Scaler


class GraphModel:
    """
    Composition of keras model and converter class for transfering structure
    object to input tensors. We add methods to train the model from
    (structures, targets) pairs
    """

    def __init__(
        self,
        model: Model,
        graph_converter: StructureGraph,
        target_scaler: Scaler = DummyScaler(),
        metadata: dict | None = None,
        **kwargs,
    ):
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

    def train(
        self,
        train_structures: list[Structure],
        train_targets: list[float],
        validation_structures: list[Structure] | None = None,
        validation_targets: list[float] | None = None,
        sample_weights: list[float] | None = None,
        epochs: int = 1000,
        batch_size: int = 128,
        verbose: int = 1,
        callbacks: list[Callback] | None = None,
        scrub_failed_structures: bool = False,
        prev_model: str | None = None,
        save_checkpoint: bool = True,
        patience: int = 500,
        dirname: str = "callback",
        **kwargs,
    ) -> GraphModel:
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
            patience: (int) patience for early stopping
            dirname: (str) the directory in which to save checkpoints, if `save_checkpoint=True`
            **kwargs:
        """
        train_graphs, train_targets = self.get_all_graphs_targets(
            train_structures, train_targets, scrub_failed_structures=scrub_failed_structures
        )
        if (validation_structures is not None) and (validation_targets is not None):
            val_graphs, validation_targets = self.get_all_graphs_targets(
                validation_structures, validation_targets, scrub_failed_structures=scrub_failed_structures
            )
        else:
            val_graphs = None

        self.train_from_graphs(
            train_graphs,
            train_targets,
            validation_graphs=val_graphs,
            validation_targets=validation_targets,
            sample_weights=sample_weights,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
            prev_model=prev_model,
            patience=patience,
            save_checkpoint=save_checkpoint,
            dirname=dirname,
            **kwargs,
        )
        return self

    def train_from_graphs(
        self,
        train_graphs: list[dict],
        train_targets: list[float],
        validation_graphs: list[dict] | None = None,
        validation_targets: list[float] | None = None,
        sample_weights: list[float] | None = None,
        epochs: int = 1000,
        batch_size: int = 128,
        verbose: int = 1,
        callbacks: list[Callback] | None = None,
        prev_model: str | None = None,
        patience: int = 500,
        save_checkpoint: bool = True,
        dirname: str = "callback",
        **kwargs,
    ) -> GraphModel:
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
            patience: (int) patience for early stopping
            save_checkpoint: (bool) whether to save checkpoint
            dirname: (str) the directory in which to save checkpoints, if `save_checkpoint=True`
            **kwargs:
        """
        # load from saved model
        if prev_model:
            self.load_weights(prev_model)
        is_classification = "entropy" in str(self.model.loss)
        monitor = "val_acc" if is_classification else "val_mae"
        mode = "max" if is_classification else "min"
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if callbacks is None:
            callbacks = []
        # with this call back you can stop the model training by `touch STOP`
        callbacks.append(ManualStop())
        train_nb_atoms = [len(i["atom"]) for i in train_graphs]
        train_targets = [self.target_scaler.transform(i, j) for i, j in zip(train_targets, train_nb_atoms)]
        if (validation_graphs is not None) and (validation_targets is not None):
            filepath = os.path.join(dirname, f"{monitor}_{{epoch:05d}}_{{{monitor}:.6f}}.hdf5")
            val_nb_atoms = [len(i["atom"]) for i in validation_graphs]
            validation_targets = [self.target_scaler.transform(i, j) for i, j in zip(validation_targets, val_nb_atoms)]
            val_inputs = self.graph_converter.get_flat_data(validation_graphs, validation_targets)

            val_generator = self._create_generator(*val_inputs, batch_size=batch_size)
            steps_per_val = int(np.ceil(len(validation_graphs) / batch_size))
            if save_checkpoint:
                callbacks.append(
                    ModelCheckpointMAE(
                        filepath=filepath,
                        monitor=monitor,
                        mode=mode,
                        save_best_only=True,
                        save_weights_only=False,
                        val_gen=val_generator,
                        steps_per_val=steps_per_val,
                        target_scaler=self.target_scaler,
                    )
                )
                val_generator = None  # type: ignore
                steps_per_val = None  # type: ignore

                if patience is not None:
                    callbacks.append(
                        EarlyStopping(
                            filepath=filepath,
                            monitor=monitor,
                            mode=mode,
                            patience=patience,
                        )
                    )
        else:
            val_generator = None  # type: ignore
            steps_per_val = None  # type: ignore

        train_inputs = self.graph_converter.get_flat_data(train_graphs, train_targets)
        # check dimension match
        self.check_dimension(train_graphs[0])
        train_generator = self._create_generator(*train_inputs, sample_weights=sample_weights, batch_size=batch_size)
        steps_per_train = int(np.ceil(len(train_graphs) / batch_size))
        self.fit(
            train_generator,
            steps_per_epoch=steps_per_train,
            validation_data=val_generator,
            validation_steps=steps_per_val,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            **kwargs,
        )
        return self

    def check_dimension(self, graph: dict) -> bool:
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

        for i, j, k in zip(
            ["atom features", "bond features", "state features"], input_shapes[:3], model_input_shapes[:3]
        ):
            matched = _check_match(j, k)
            if not matched:
                raise ValueError(f"The data dimension for {i} is {j} and does not match model required shape of {k}")
        return False

    def get_all_graphs_targets(
        self, structures: list[Structure], targets: list[float], scrub_failed_structures: bool = False
    ) -> tuple:
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
                    warn(f"structure with index {i} failed the graph computations", UserWarning)
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

    def predict_structures(self, structures: list[Structure], batch_size: int = 128, pbar: bool = False) -> np.ndarray:
        """
        Predict properties of structure list

        Args:
            structures: list of pymatgen Structure/Molecule

        Returns:
            predicted target values
        """
        graphs = [self.graph_converter.convert(structure) for structure in structures]
        return self.predict_graphs(graphs, batch_size=batch_size, pbar=pbar)

    def predict_graph(self, graph: dict) -> np.ndarray:
        """
        Predict property from graph

        Args:
            graph: a graph dictionary, see megnet.data.graph

        Returns:
            predicted target value

        """
        inp = self.graph_converter.graph_to_input(graph)
        pred = self.predict(inp, verbose=False)  # direct prediction, shape [1, 1, m]
        return self.target_scaler.inverse_transform(pred[0, 0], len(graph["atom"]))

    def predict_graphs(self, graphs: list[dict], batch_size: int = 128, pbar: bool = False) -> np.ndarray:
        """
        Predict properties from graphs

        Args:
            graphs: a list graph dictionary, see megnet.data.graph

        Returns:
            predicted target values

        """
        inputs = self.graph_converter.get_flat_data(graphs)
        n_atoms = [len(graph["atom"]) for graph in graphs]
        pred_gen = self._create_generator(*inputs, batch_size=batch_size, is_shuffle=False)
        predicted = []
        if pbar:
            pred_gen = tqdm(pred_gen, total=len(pred_gen))
        for i in pred_gen:
            predicted.append(self.predict(i, verbose=False))
        pred_targets = np.concatenate(predicted, axis=1)[0]
        return np.array([self.target_scaler.inverse_transform(i, j) for i, j in zip(pred_targets, n_atoms)])

    def _create_generator(self, *args, **kwargs) -> GraphBatchDistanceConvert | GraphBatchGenerator:
        if hasattr(self.graph_converter, "bond_converter"):
            kwargs.update({"distance_converter": self.graph_converter.bond_converter})
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
            {"graph_converter": self.graph_converter, "target_scaler": self.target_scaler, "metadata": self.metadata},
            filename + ".json",
        )

    @classmethod
    def from_file(cls, filename: str) -> GraphModel:
        """
        Class method to load model from
            filename for keras model
            filename.json for additional converters

        Args:
            filename: (str) model file name

        Returns
            GraphModel
        """
        configs = loadfn(filename + ".json")
        from tensorflow.keras.models import load_model

        from megnet.layers import _CUSTOM_OBJECTS

        model = load_model(filename, custom_objects=_CUSTOM_OBJECTS)
        configs.update({"model": model})
        return GraphModel(**configs)
