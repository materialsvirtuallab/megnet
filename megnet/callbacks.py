from keras.callbacks import Callback
import keras.backend as K
from megnet.utils.metric_utils import mae, accuracy
import numpy as np
import os
import warnings
from glob import glob
from collections import deque


class GeneratorLog(Callback):
    """
    This callback print out the MAE for train_generator and validation_generator every n_every steps.
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
            self.yscaler = _DummyScaler()

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
            print("Train MAE")
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
            print("Test MAE")
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
        y_scaler: (object) exposing inverse_transform method to scale the output
        period: (int) number of epoch interval for this callback
        is_pa: (bool) if it is a per-atom quantity
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
                 y_scaler=None,
                 period=1,
                 is_pa=False,
                 mode='auto'):
        super().__init__()
        if val_gen is None:
            raise ValueError('No validation data is provided!')
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.val_gen = val_gen
        self.steps_per_val = steps_per_val
        self.yscaler = y_scaler
        self.is_pa = is_pa
        if self.yscaler is None:
            self.yscaler = _DummyScaler()

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
                if not self.is_pa:
                    nb_atom = np.ones_like(nb_atom)
                pred_ = self.model.predict(val_data[0])
                val_pred.append(self.yscaler.inverse_transform(pred_[0, :, :]) * nb_atom[:, None])
                val_y.append(self.yscaler.inverse_transform(val_data[1][0, :, :]) * nb_atom[:, None])
            current = self.metric(np.concatenate(val_y, axis=0), np.concatenate(val_pred, axis=0))
            filepath = self.filepath.format(**{"epoch": epoch + 1, self.monitor: current})

            if self.save_best_only:
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % self.monitor, RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
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
        callback_dir: (str) the directory where the saved models are stored
        factor: (float) a value < 1 for scaling the learning rate
        verbose: (int) whether to show the loading event
        patience: (int) number of steps that the val mae does not change. It is a criteria for early stopping
        epoch_local_in_fname: (int) combine with splitter to find the epoch number in file name
        splitter: (str) a splitter string for the file name, e.g., callback/val_mae_0013_0.123.hdf5, can use _ as
        splitter and epoch_local_in_fname of 2 to get the correct epoch number of 13.
    """

    def __init__(self,
                 callback_dir='./callback',
                 factor=0.5,
                 verbose=1,
                 patience=500,
                 epoch_local_in_fname=2,
                 splitter='_'):

        self.callback_dir = callback_dir
        self.verbose = verbose
        self.factor = factor
        self.losses = deque([], maxlen=10)
        self.patience = patience
        self.epoch_local_in_fname = epoch_local_in_fname
        self.splitter = splitter

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')

        files = glob(os.path.join(self.callback_dir, "*"))
        if len(files) > 1:
            latest_file = max(files, key=lambda x: int(x.split(self.splitter)[self.epoch_local_in_fname]))
        else:
            latest_file = None
        if latest_file:
            last_saved_epoch = int(latest_file.split(self.splitter)[self.epoch_local_in_fname])
            if last_saved_epoch + self.patience <= epoch:
                self.model.stop_training = True
                print('MAE does not improve after %d, stopping the fitting...' % self.patience)

        if loss is not None:
            self.losses.append(loss)
            if np.isnan(loss) or np.isinf(loss):
                self._reduce_lr_and_load()
                if self.verbose:
                    print("Nan loss found!\n")
                    print("Now lr is ", float(K.get_value(self.model.optimizer.lr)), "\n")
            else:
                if len(self.losses) > 1:
                    if self.losses[-1] > (self.losses[-2] * 100):
                        self._reduce_lr_and_load()
                        if self.verbose:
                            print("Loss shoot up from %.3f to %.3f! Reducing lr \n" %(self.losses[-1], self.losses[-2]))
                            print("Now lr is ", float(K.get_value(self.model.optimizer.lr)), "\n")

    def _reduce_lr_and_load(self):
        old_value = float(K.get_value(self.model.optimizer.lr))
        files = glob(os.path.join(self.callback_dir, "*"))
        if len(files) > 1:
            latest_file = max(files, key=os.path.getctime)
        else:
            latest_file = None

        self.model.reset_states()
        K.set_value(self.model.optimizer.lr, old_value*self.factor)
        opt_dict = self.model.optimizer.get_config()
        self.model.compile(self.model.optimizer.__class__(**opt_dict), self.model.loss)
        if latest_file is not None:
            self.model.load_weights(latest_file)
            if self.verbose:
                print("Load weights %s" % latest_file)


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
    print(' '.join(line))
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


class _DummyScaler(object):
    """
    Does nothing
    """
    def inverse_transform(self, x):
        return x
