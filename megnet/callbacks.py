import re
import logging
import os
import warnings
from glob import glob
from collections import deque
import numpy as np
from keras.callbacks import Callback
import keras.backend as kb
from megnet.utils.metrics import mae, accuracy
from megnet.utils.preprocessing import DummyScaler

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GeneratorLog(Callback):
    """
    This callback logger.info out the MAE for train_generator and validation_generator every n_every steps.
    The default keras training log does not contain method to rescale the results, thus is not physically
    intuitive.

    Args:
        train_gen: (generator), yield (x, y) pairs for training
        steps_per_train: (int) number of generator steps per training epoch
        val_gen: (generator), yield (x, y) pairs for validation.
        steps_per_val: (int) number of generator steps per epoch for validation data
        y_scaler: (object) y_scaler.inverse_transform is used to convert the predicted values to its original scale
        n_every: (int) epoch interval for showing the log
        val_names: (list of string) variable names
        val_units: (list of string) variable units
        is_pa: (bool) whether it is a per-atom quantity
    """

    def __init__(self, train_gen, steps_per_train=None,
                 val_gen=None, steps_per_val=None, y_scaler=None, n_every=5,
                 val_names=None, val_units=None, is_pa=False):
        super().__init__()
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.steps_per_train = steps_per_train
        self.steps_per_val = steps_per_val
        self.yscaler = y_scaler
        self.epochs = []
        self.total_epoch = 0
        self.n_every = n_every
        self.val_names = val_names
        self.val_units = val_units
        self.is_pa = is_pa
        if self.yscaler is None:
            self.yscaler = DummyScaler()

    def on_epoch_end(self, epoch, logs=None):
        """
        Standard keras callback interface, executed at the end of epoch
        """
        self.total_epoch += 1
        if self.total_epoch % self.n_every == 0:
            train_pred = []
            train_y = []
            for i in range(self.steps_per_train):
                train_data = self.train_gen[i]
                nb_atom = _count(np.array(train_data[0][-2]))
                if not self.is_pa:
                    nb_atom = np.ones_like(nb_atom)
                pred_ = self.model.predict(train_data[0])
                train_pred.append(self.yscaler.inverse_transform(pred_[0, :, :]) * nb_atom[:, None])
                train_y.append(self.yscaler.inverse_transform(train_data[1][0, :, :]) * nb_atom[:, None])
            train_mae = np.mean(np.abs(np.concatenate(train_pred, axis=0) - np.concatenate(train_y, axis=0)), axis=0)
            logger.info("Train MAE")
            _print_mae(self.val_names, train_mae, self.val_units)
            val_pred = []
            val_y = []
            for i in range(self.steps_per_val):
                val_data = self.val_gen[i]
                nb_atom = _count(np.array(val_data[0][-2]))
                if not self.is_pa:
                    nb_atom = np.ones_like(nb_atom)
                pred_ = self.model.predict(val_data[0])
                val_pred.append(self.yscaler.inverse_transform(pred_[0, :, :]) * nb_atom[:, None])
                val_y.append(self.yscaler.inverse_transform(val_data[1][0, :, :]) * nb_atom[:, None])
            val_mae = np.mean(np.abs(np.concatenate(val_pred, axis=0) - np.concatenate(val_y, axis=0)), axis=0)
            logger.info("Test MAE")
            _print_mae(self.val_names, val_mae, self.val_units)
            self.model.history.history.setdefault("train_mae", []).append(train_mae)
            self.model.history.history.setdefault("val_mae", []).append(val_mae)


class ModelCheckpointMAE(Callback):
    """
    Save the best MAE model

    Args:
        filepath: (string) path to save the model file with format. For example
        `weights.{epoch:02d}-{val_mae:.6f}.hdf5` will save the corresponding epoch and val_mae in the filename
        monitor: (string) quantity to monitor, default to "val_mae"
        verbose: (int) 0 for no training log, 1 for only epoch-level log and 2 for batch-level log
        save_best_only: (bool) whether to save only the best model
        save_weights_only: (bool) whether to save the weights only excluding model structure
        val_gen: (generator) validation generator
        steps_per_val: (int) steps per epoch for validation generator
        target_scaler: (object) exposing inverse_transform method to scale the output
        period: (int) number of epoch interval for this callback
        mode: (string) choose from "min", "max" or "auto"
    """

    def __init__(self,
                 filepath='./callback/val_mae_{epoch:05d}_{val_mae:.6f}.hdf5',
                 monitor='val_mae',
                 verbose=0,
                 save_best_only=True,
                 save_weights_only=False,
                 val_gen=None,
                 steps_per_val=None,
                 target_scaler=None,
                 period=1,
                 mode='auto'):
        super().__init__()
        if val_gen is None:
            raise ValueError('No validation data is provided!')
        self.verbose = verbose
        if self.verbose > 0:
            logging.basicConfig(level=logging.INFO)
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.val_gen = val_gen
        self.steps_per_val = steps_per_val
        self.target_scaler = target_scaler
        if self.target_scaler is None:
            self.target_scaler = DummyScaler()

        if monitor == 'val_mae':
            self.metric = mae
            self.monitor = 'val_mae'
        elif monitor == 'val_acc':
            self.metric = accuracy
            self.filepath = self.filepath.replace('val_mae', 'val_acc')
            self.monitor = 'val_acc'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            val_pred = []
            val_y = []
            for i in range(self.steps_per_val):
                val_data = self.val_gen[i]
                nb_atom = _count(np.array(val_data[0][-2]))
                pred_ = self.model.predict(val_data[0])
                val_pred.append(self.target_scaler.inverse_transform(pred_[0, :, :], nb_atom[:, None]))
                val_y.append(self.target_scaler.inverse_transform(val_data[1][0, :, :], nb_atom[:, None]))
            current = self.metric(np.concatenate(val_y, axis=0), np.concatenate(val_pred, axis=0))
            filepath = self.filepath.format(**{"epoch": epoch + 1, self.monitor: current})

            if self.save_best_only:
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % self.monitor, RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        logger.info('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                    ' saving model to %s' % (epoch + 1, self.monitor, self.best, current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            logger.info('\nEpoch %05d: %s did not improve from %0.5f' %
                                        (epoch + 1, self.monitor, self.best))
            else:
                logger.info('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


class ManualStop(Callback):
    """
    Stop the training manually by putting a "STOP" file in the directory
    """

    def on_batch_end(self, epoch, logs=None):
        if os.path.isfile('STOP'):
            self.model.stop_training = True


class ReduceLRUponNan(Callback):
    """
    This callback function solves a problem that when doing regression, an nan loss may occur, or the
    loss suddenly shoot up
    If such things happen, the model will reduce the learning rate and load the last best model during the
    training process.
    It has an extra function that patience for early stopping. This will move to indepedent callback in the
    future.

    Args:
        filepath: (str) filepath for saved model checkpoint, should be consistent with checkpoint callback
        factor: (float) a value < 1 for scaling the learning rate
        verbose: (int) whether to show the loading event
        patience: (int) number of steps that the val mae does not change. It is a criteria for early stopping
        track_metric: (string) the variable to track
    """

    def __init__(self,
                 filepath='./callback/val_mae_{epoch:05d}_{val_mae:.6f}.hdf5',
                 factor=0.5,
                 verbose=1,
                 patience=500,
                 monitor='val_mae',
                 mode='auto'):
        self.filepath = filepath
        self.verbose = verbose
        self.factor = factor
        self.losses = deque([], maxlen=10)
        self.patience = patience
        self.monitor = monitor
        super().__init__()

        if mode == 'min':
            self.monitor_op = np.argmin
        elif mode == 'max':
            self.monitor_op = np.argmax
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.argmax
            else:
                self.monitor_op = np.argmin

        # get variable name
        variable_name_pattern = r'{(.+?)}'
        self.variable_names = re.findall(variable_name_pattern, filepath)
        self.variable_names = [i.split(':')[0] for i in self.variable_names]
        if self.monitor not in self.variable_names:
            raise ValueError("The monitored metric should be in the name pattern")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        last_saved_epoch, last_metric, last_file = self._get_checkpoints()
        if last_saved_epoch is not None:
            if last_saved_epoch + self.patience <= epoch:
                self.model.stop_training = True
                logger.info('%s does not improve after %d, stopping the fitting...' % (self.monitor, self.patience))

        if loss is not None:
            self.losses.append(loss)
            if np.isnan(loss) or np.isinf(loss):
                if self.verbose:
                    logger.info("Nan loss found!")
                self._reduce_lr_and_load(last_file)
                if self.verbose:
                    logger.info("Now lr is %s." % float(kb.get_value(self.model.optimizer.lr)))
            else:
                if len(self.losses) > 1:
                    if self.losses[-1] > (self.losses[-2] * 100):
                        self._reduce_lr_and_load(last_file)
                        if self.verbose:
                            logger.info(
                                "Loss shot up from %.3f to %.3f! Reducing lr " % (self.losses[-1], self.losses[-2]))
                            logger.info("Now lr is %s." % float(kb.get_value(self.model.optimizer.lr)))

    def _reduce_lr_and_load(self, last_file):
        old_value = float(kb.get_value(self.model.optimizer.lr))
        self.model.reset_states()
        kb.set_value(self.model.optimizer.lr, old_value * self.factor)
        opt_dict = self.model.optimizer.get_config()
        self.model.compile(self.model.optimizer.__class__(**opt_dict), self.model.loss)
        if last_file is not None:
            self.model.load_weights(last_file)
            if self.verbose:
                logger.info("Load weights %s" % last_file)
        else:
            logger.info("No weights were loaded")

    def _get_checkpoints(self):
        file_pattern = re.sub('{(.+?)}', '([0-9\.]+)', self.filepath)
        glob_pattern = re.sub('{(.+?)}', '*', self.filepath)
        all_check_points = glob(glob_pattern)

        if len(all_check_points) > 0:
            metric_index = self.variable_names.index(self.monitor)
            epoch_index = self.variable_names.index('epoch')
            metric_values = []
            epochs = []
            for i in all_check_points:
                metrics = re.findall(file_pattern, i)[0]
                metric_values.append(float(metrics[metric_index]))
                epochs.append(int(metrics[epoch_index]))
            ind = self.monitor_op(metric_values)
            return epochs[ind], metric_values[ind], all_check_points[ind]
        else:
            return None, None, None


def _print_mae(target_names, maes, units):
    """
    format printing the MAE for each variable

    Args:
        target_names: (list of string) variable names
        maes:  (list of numeric) MAE values for each variable
        units:  (list of string) units for each variable

    Returns:
          bool
    """
    line = []
    for i, j, k in zip(target_names, maes, units):
        line.append(i + ': ' + '%.4f' % j + ' ' + k)
    logger.info(' '.join(line))
    return True


def _count(a):
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
