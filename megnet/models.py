from keras.optimizers import Adam
from keras.layers import Dense, Input, Concatenate, Add, Embedding, Dropout
from megnet.layers import MEGNetLayer, Set2Set
from megnet.activations import softplus2
from megnet.losses import mse_scale
from keras.regularizers import l2

from keras.models import Model
from megnet.callbacks import ModelCheckpointMAE, ManualStop
from megnet.utils.general_utils import expand_1st
from megnet.data.graph import GraphBatchDistanceConvert, GraphBatchGenerator, GaussianDistance
from megnet.data.crystal import graphs2inputs, CrystalGraph
import numpy as np
import os

from monty.serialization import dumpfn, loadfn


pjoin = os.path.join


class GraphModel:
    """
    Wrapper of keras.Model.
    We add methods to train the model from (structures, targets) pairs

    Args:
        model: (keras model)
        graph_convertor: (object) a object that turns a structure to a graph
        distance_convertor: (object) expand the distance value to a vector

    Or one can specify the maximum radius (max_r), number of basis (n_basis) and
    the Gaussian width (width) for constructing the distance convertor
    """

    def __init__(self,
                 model,
                 graph_convertor,
                 distance_convertor):
        self.model = model
        self.graph_convertor = graph_convertor
        self.distance_convertor = distance_convertor

    def __getattr__(self, p):
        return getattr(self.model, p)

    def train(self,
              train_structures,
              train_targets,
              validation_structures=None,
              validation_targets=None,
              epochs=1000,
              batch_size=128,
              verbose=1,
              callbacks=None,
              prev_model=None,
              **kwargs):
        """
        :param train_structures: (list) list of pymatgen structures
        :param train_targets: (list) list of target values
        :param validation_structures: (list) list of pymatgen structures as
            validation
        :param validation_targets: (list) list of validation targets
        :param epochs: (int) number of epochs
        :param batch_size: (int) training batch size
        :param verbose: (int) keras fit verbose, 0 no progress bar, 1 only at
            the epoch end and 2 every batch
        :param callbacks: (list) megnet or keras callback functions for training
        :param prev_model: (str) file name for previously saved model
        :param kwargs:
        :return:
        """
        train_graphs = [self.graph_convertor(i) for i in train_structures]
        if validation_structures is not None:
            val_graphs = [self.graph_convertor(i) for i in validation_structures]
        else:
            val_graphs = None

        self.train_from_graphs(train_graphs,
                               train_targets,
                               validation_graphs=val_graphs,
                               validation_targets=validation_targets,
                               epochs=epochs,
                               batch_size=batch_size,
                               verbose=verbose,
                               callbacks=callbacks,
                               prev_model=prev_model,
                               **kwargs
                               )

    def train_from_graphs(self,
                          train_graphs,
                          train_targets,
                          validation_graphs=None,
                          validation_targets=None,
                          epochs=1000,
                          batch_size=128,
                          verbose=1,
                          callbacks=None,
                          prev_model=None,
                          **kwargs
                          ):

        # load from saved model
        if prev_model:
            self.load_weights(prev_model)
        dirname = kwargs.get('dirname', 'callback')
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if callbacks is None:
            callbacks = [ManualStop()]
        train_targets = np.array(train_targets).ravel()
        if validation_graphs is not None:
            filepath = pjoin(dirname, 'val_mae_{epoch:05d}_{val_mae:.6f}.hdf5')
            validation_targets = np.array(validation_targets).ravel()
            val_inputs = graphs2inputs(validation_graphs, validation_targets)

            val_generator = self._create_generator(*val_inputs,
                                                   batch_size=batch_size)
            steps_per_val = int(np.ceil(len(validation_graphs) / batch_size))
            # print(steps_per_val)
            callbacks.extend([ModelCheckpointMAE(filepath=filepath,
                                                 save_best_only=True,
                                                 save_weights_only=False,
                                                 val_gen=val_generator,
                                                 steps_per_val=steps_per_val,
                                                 y_scaler=None)])
        else:
            val_generator = None
            steps_per_val = None
        train_inputs = graphs2inputs(train_graphs, train_targets)
        train_generator = self._create_generator(*train_inputs, batch_size=batch_size)
        steps_per_train = int(np.ceil(len(train_graphs) / batch_size))
        self.fit_generator(train_generator, steps_per_epoch=steps_per_train,
                           validation_data=val_generator, validation_steps=steps_per_val,
                           epochs=epochs, verbose=verbose, callbacks=callbacks)

    def predict_structure(self, structure):
        """
        Predict property from structure
        :param structure:
        :return:
        """
        graph = self.graph_convertor(structure)
        gnode = [0] * len(graph['node'])
        gbond = [0] * len(graph['index1'])
        inp = [expand_1st(graph['node']),
               expand_1st(self.distance_convertor.convert(graph['distance'])),
               expand_1st(np.array(graph['state'])),
               expand_1st(np.array(graph['index1'])),
               expand_1st(np.array(graph['index2'])),
               expand_1st(np.array(gnode)),
               expand_1st(np.array(gbond)),
               ]
        return self.predict(inp).ravel()

    def _create_generator(self, *args, **kwargs):
        if self.distance_convertor is not None:
            kwargs.update({'distance_convertor': self.distance_convertor})
            return GraphBatchDistanceConvert(*args, **kwargs)
        else:
            return GraphBatchGenerator(*args, **kwargs)

    def save_model(self, filename):
        """
        Save the model to a keras model hdf5 and a json config for additional
        convertors

        :param filename: (str)
        :return:
        """
        self.model.save(filename)
        dumpfn(
            {
                'distance_convertor': self.distance_convertor,
                'graph_convertor': self.graph_convertor
            },
            filename + '.json'
        )

    @classmethod
    def from_file(cls, filename):
        """
        Class method to load model from
            filename for keras model
            filename.json for additional convertors

        :param filename: (str)
        :return: Model
        """
        configs = loadfn(filename + '.json')
        from keras.utils import get_custom_objects
        from keras.models import load_model

        custom_objs = get_custom_objects()
        custom_objs.update({'mean_squared_error_with_scale': mse_scale,
                            'softplus2': softplus2,
                            'MEGNetLayer': MEGNetLayer,
                            'Set2Set': Set2Set})
        model = load_model(filename, custom_objects=custom_objs)
        configs.update({'model': model})
        return GraphModel(**configs)


class MEGNetModel(GraphModel):

    def __init__(self,
                 nfeat_edge,
                 nfeat_global,
                 nfeat_node=None,
                 nblocks=3,
                 lr=1e-3,
                 n1=64,
                 n2=32,
                 n3=16,
                 nvocal=95,
                 embedding_dim=16,
                 npass=3,
                 ntarget=1,
                 act=softplus2,
                 is_classification=False,
                 loss="mse",
                 l2_coef=None,
                 dropout=None,
                 graph_convertor=None,
                 distance_convertor=None):
        """
        Construct a graph network model with or without explicit atom features
        if n_feature is specified then a general graph model is assumed,
        otherwise a crystal graph model with z number as atom feature is assumed.

        :param nfeat_edge: (int) number of bond features
        :param nfeat_global: (int) number of state features
        :param nfeat_node: (int) number of atom features
        :param nblocks: (int) number of MEGNetLayer blocks
        :param lr: (float) learning rate
        :param n1: (int) number of hidden units in layer 1 in MEGNetLayer
        :param n2: (int) number of hidden units in layer 2 in MEGNetLayer
        :param n3: (int) number of hidden units in layer 3 in MEGNetLayer
        :param nvocal: (int) number of total element
        :param embedding_dim: (int) number of embedding dimension
        :param npass: (int) number of recurrent steps in Set2Set layer
        :param ntarget: (int) number of output targets
        :param act: (object) activation function
        :param l2_coef: (float or None) l2 regularization parameter
        :param is_classification: (bool) whether it is a classifiation task
        :param loss: (object or str) loss function
        :param dropout: (float) dropout rate
        :param graph_convertor: (object) object that exposes a "convert" method
            for structure to graph conversion
        :param distance_convertor: (object) object that exposes a "convert"
            method for distance to expanded vector conversion
        :return: keras model object
        """
        int32 = 'int32'
        if nfeat_node is None:
            x1 = Input(shape=(None,), dtype=int32)  # only z as feature
            x1_ = Embedding(nvocal, embedding_dim)(x1)
        else:
            x1 = Input(shape=(None, nfeat_node))
            x1_ = x1
        x2 = Input(shape=(None, nfeat_edge))
        x3 = Input(shape=(None, nfeat_global))
        x4 = Input(shape=(None,), dtype=int32)
        x5 = Input(shape=(None,), dtype=int32)
        x6 = Input(shape=(None,), dtype=int32)
        x7 = Input(shape=(None,), dtype=int32)

        if l2_coef is not None:
            reg = l2(l2_coef)
        else:
            reg = None

        # two feedforward layers
        def ff(x, n_hiddens=[n1, n2]):
            out = x
            for i in n_hiddens:
                out = Dense(i, activation=act, kernel_regularizer=reg)(out)
            return out

        # a block corresponds to two feedforward layers + one MEGNetLayer layer
        # Note the first block does not contain the feedforward layer since
        # it will be explicitly added before the block
        def one_block(a, b, c, has_ff=True):
            if has_ff:
                x1_ = ff(a)
                x2_ = ff(b)
                x3_ = ff(c)
            else:
                x1_ = a
                x2_ = b
                x3_ = c
            out = MEGNetLayer(
                [n1, n1, n2], [n1, n1, n2], [n1, n1, n2],
                pool_method='mean', activation=act)(
                [x1_, x2_, x3_, x4, x5, x6, x7])

            x1_temp = out[0]
            x2_temp = out[1]
            x3_temp = out[2]
            if dropout:
                x1_temp = Dropout(dropout)(x1_temp)
                x2_temp = Dropout(dropout)(x2_temp)
                x3_temp = Dropout(dropout)(x3_temp)
            return x1_temp, x2_temp, x3_temp

        x1_ = ff(x1_)
        x2_ = ff(x2)
        x3_ = ff(x3)
        for i in range(nblocks):
            if i == 0:
                has_ff = False
            else:
                has_ff = True
            x1_1 = x1_
            x2_1 = x2_
            x3_1 = x3_
            x1_1, x2_1, x3_1 = one_block(x1_1, x2_1, x3_1, has_ff)
            # skip connection
            x1_ = Add()([x1_, x1_1])
            x2_ = Add()([x2_, x2_1])
            x3_ = Add()([x3_, x3_1])

        # set2set for both the atom and bond
        node_vec = Set2Set(T=npass, n_hidden=n3)([x1_, x6])
        edge_vec = Set2Set(T=npass, n_hidden=n3)([x2_, x7])
        # concatenate atom, bond, and global
        final_vec = Concatenate(axis=-1)([node_vec, edge_vec, x3_])
        if dropout:
            final_vec = Dropout(dropout)(final_vec)
        # final dense layers
        final_vec = Dense(n2, activation=act)(final_vec)
        final_vec = Dense(n3, activation=act)(final_vec)

        if is_classification:
            final_act = 'sigmoid'
            loss = 'binary_crossentropy'
        else:
            final_act = None
            loss = loss

        out = Dense(ntarget, activation=final_act)(final_vec)
        model = Model(inputs=[x1, x2, x3, x4, x5, x6, x7], outputs=out)
        model.compile(Adam(lr), loss)

        if graph_convertor is None:
            graph_convertor = CrystalGraph()
        if distance_convertor is None:
            distance_convertor = GaussianDistance(np.linspace(0, 5, 100), 0.5)
        super(MEGNetModel, self).__init__(
            model=model, graph_convertor=graph_convertor,
            distance_convertor=distance_convertor)
