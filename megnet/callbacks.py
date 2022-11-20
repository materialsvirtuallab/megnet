"""
callbacks functions used in training process
"""
from __future__ import annotations

import logging
import os
import re
import warnings
from collections import deque
from glob import glob

import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import Sequence

from megnet.utils.metrics import accuracy, mae
from megnet.utils.preprocessing import DummyScaler, Scaler

logger = logging.getLogger(__name__)


class ModelCheckpointMAE(Callback):
    """
    Save the best MAE model with target scaler
    """

    def __init__(
        self,
        filepath: str = "./callback/val_mae_{epoch:05d}_{val_mae:.6f}.hdf5",
        monitor: str = "val_mae",
        verbose: int = 0,
        save_best_only: bool = True,
        save_weights_only: bool = False,
        val_gen: Sequence = None,
        steps_per_val: int | None = None,
        target_scaler: Scaler | None = None,
        period: int = 1,
        mode: str = "auto",
    ):
        """
        Args:
            filepath (string): path to save the model file with format. For example
                `weights.{epoch:02d}-{val_mae:.6f}.hdf5` will save the corresponding epoch and
                val_mae in the filename
            monitor (string): quantity to monitor, default to "val_mae"
            verbose (int): 0 for no training log, 1 for only epoch-level log and 2 for batch-level log
            save_best_only (bool): whether to save only the best model
            save_weights_only (bool): whether to save the weights only excluding model structure
            val_gen (generator): validation generator
            steps_per_val (int): steps per epoch for validation generator
            target_scaler (object): exposing inverse_transform method to scale the output
            period (int): number of epoch interval for this callback
            mode: (string) choose from "min", "max" or "auto"
        """
        super().__init__()
        if val_gen is None:
            raise ValueError("No validation data is provided!")
        self.verbose = verbose
        if self.verbose > 0:
            logging.basicConfig(level=logging.INFO)
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.val_gen = val_gen
        self.steps_per_val = steps_per_val or len(val_gen)
        self.target_scaler = target_scaler or DummyScaler()

        if monitor == "val_mae":
            self.metric = mae
            self.monitor = "val_mae"
        elif monitor == "val_acc":
            self.metric = accuracy
            self.filepath = self.filepath.replace("val_mae", "val_acc")
            self.monitor = "val_acc"

        if mode == "min":
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == "max":
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if "acc" in self.monitor or self.monitor.startswith("fmeasure"):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """
        Codes called by the callback at the end of epoch
        Args:
            epoch (int): epoch id
            logs (dict): logs of training

        Returns:
            None
        """
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            val_pred = []
            val_y = []
            for i in range(self.steps_per_val):
                val_data = self.val_gen[i]  # type: ignore
                nb_atom = _count(np.array(val_data[0][-2]))
                stop_training = self.model.stop_training  # save stop_trainings state
                pred_ = self.model.predict(val_data[0])
                self.model.stop_training = stop_training
                val_pred.append(self.target_scaler.inverse_transform(pred_[0, :, :], nb_atom[:, None]))
                val_y.append(self.target_scaler.inverse_transform(val_data[1][0, :, :], nb_atom[:, None]))
            current = self.metric(np.concatenate(val_y, axis=0), np.concatenate(val_pred, axis=0))
            filepath = self.filepath.format(**{"epoch": epoch + 1, self.monitor: current})

            if self.save_best_only:
                if current is None:
                    warnings.warn(f"Can save best model only with {self.monitor} available, skipping.", RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        logger.info(
                            f"\nEpoch {epoch+1:05d}: {self.monitor} improved from {self.best:.5f} to {current:.5f},"
                            f" saving model to {filepath}"
                        )
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            logger.info(f"\nEpoch {epoch+1:05d}: {self.monitor} did not improve from {self.best:.5f}")
            else:
                logger.info(f"\nEpoch {epoch+1:05d}: saving model to {filepath}")
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


class ManualStop(Callback):
    """
    Stop the training manually by putting a "STOP" file in the directory
    """

    def on_batch_end(self, epoch: int, logs: dict | None = None) -> None:
        """
        Codes called at the end of a batch
        Args:
            epoch (int): epoch id
            logs (dict): log dict

        Returns: None

        """
        if os.path.isfile("STOP"):
            self.model.stop_training = True


class EarlyStopping(Callback):
    """
    Implements EarlyStopping callback using saved model files
    """

    def __init__(
        self,
        filepath: str = "./callback/val_mae_{epoch:05d}_{val_mae:.6f}.hdf5",
        patience: int = 500,
        monitor: str = "val_mae",
        mode: str = "auto",
    ):
        """
        Args:
            filepath (str): filepath for saved model checkpoint, should be consistent with
                checkpoint callback
            patience (int): number of steps that the val mae does not change.
                It is a criteria for early stopping
            monitor (str): target metric to monitor
            mode (str): min, max or auto
        """
        self.filepath = filepath
        self.losses: deque = deque([], maxlen=10)
        self.patience = patience
        self.monitor = monitor
        super().__init__()

        if mode == "min":
            self.monitor_op = np.argmin
        elif mode == "max":
            self.monitor_op = np.argmax
        else:
            if "acc" in self.monitor:
                self.monitor_op = np.argmax
            else:
                self.monitor_op = np.argmin

        # get variable name
        variable_name_pattern = r"{(.+?)}"
        self.variable_names = re.findall(variable_name_pattern, filepath)
        self.variable_names = [i.split(":")[0] for i in self.variable_names]
        if self.monitor not in self.variable_names:
            raise ValueError("The monitored metric should be in the name pattern")

    def on_epoch_end(self, epoch: int, logs: dict | None = None):
        """
        Check the loss value at the end of an epoch for early stopping
        Args:
            epoch (int): epoch id
            logs (dict): log history

        Returns: None

        """
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None:
            self.losses.append(loss)
            if np.isnan(loss) or np.isinf(loss):
                logger.info("Nan loss found!")
                self.model.stop_training = True

        last_saved_epoch, last_metric, last_file = self._get_checkpoints()
        if last_saved_epoch is not None:
            if last_saved_epoch + self.patience <= epoch:
                self.model.stop_training = True
                logger.info(f"{self.monitor} does not improve after {self.patience}, stopping the fitting...")

    def _get_checkpoints(self):
        file_pattern = re.sub(r"{(.+?)}", r"([0-9\.]+)", self.filepath)
        glob_pattern = re.sub(r"{(.+?)}", r"*", self.filepath)
        all_check_points = glob(glob_pattern)

        if len(all_check_points) > 0:
            metric_index = self.variable_names.index(self.monitor)
            epoch_index = self.variable_names.index("epoch")
            metric_values = []
            epochs = []
            for i in all_check_points:
                metrics = re.findall(file_pattern, i)[0]
                metric_values.append(float(metrics[metric_index]))
                epochs.append(int(metrics[epoch_index]))
            ind = self.monitor_op(metric_values)
            return epochs[ind], metric_values[ind], all_check_points[ind]
        return None, None, None


def _count(a: np.ndarray) -> np.ndarray:
    """
    count number of appearance for each element in a

    Args:
        a: (np.array)

    Returns:
        (np.array) number of appearance of each element in a
    """
    a = a.ravel()
    a = np.r_[a[0], a, np.Inf]
    z = np.where(np.abs(np.diff(a)) > 0)[0]
    z = np.r_[0, z]
    return np.diff(z)
