from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Input, Concatenate, Add, Embedding, Dropout
from megnet.layers import MEGNet, Set2Set
from megnet.activations import softplus2
from megnet.losses import mse_scale
from keras.regularizers import l2


def set2set_model(n_feature,
                  n_connect,
                  n_global,
                  n_blocks=3,
                  lr=1e-3,
                  n1=64,
                  n2=32,
                  n3=16,
                  n_pass=3,
                  n_target=1,
                  act=softplus2,
                  dropout=None):
    """
    construct a graph network model with explicit atom features

    :param n_feature: (int) number of atom features
    :param n_connect: (int) number of bond features
    :param n_global: (int) number of state features
    :param n_blocks: (int) number of MEGNet block
    :param lr: (float) learning rate
    :param n1: (int) number of hidden units in layer 1 in MEGNet
    :param n2: (int) number of hidden units in layer 2 in MEGNet
    :param n3: (int) number of hidden units in layer 3 in MEGNet
    :param n_pass: (int) number of recurrent steps in Set2Set layer
    :param n_target: (int) number of output targets
    :param act: (object) activation function
    :param dropout: (float) dropout rate
    :return: keras model object
    """
    int32 = 'int32'
    x1 = Input(shape=(None, n_feature))
    x2 = Input(shape=(None, n_connect))
    x3 = Input(shape=(None, n_global))
    x4 = Input(shape=(None,), dtype=int32)
    x5 = Input(shape=(None,), dtype=int32)
    x6 = Input(shape=(None,), dtype=int32)
    x7 = Input(shape=(None,), dtype=int32)

    # two feedforward layers
    def ff(x, n_hiddens=[n1, n2]):
        out = x
        for i in n_hiddens:
            out = Dense(i, activation=act)(out)
        return out

    # a block corresponds to two feedforward layers + one MEGNet layer
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
        out = MEGNet([n1, n1, n2], [n1, n1, n2], [n1, n1, n2], pool_method='mean', activation=act)(
            [x1_, x2_, x3_, x4, x5, x6, x7])

        x1_temp = out[0]
        x2_temp = out[1]
        x3_temp = out[2]
        if dropout:
            x1_temp = Dropout(dropout)(x1_temp)
            x2_temp = Dropout(dropout)(x2_temp)
            x3_temp = Dropout(dropout)(x3_temp)
        return x1_temp, x2_temp, x3_temp

    x1_ = ff(x1)
    x2_ = ff(x2)
    x3_ = ff(x3)
    for i in range(n_blocks):
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
    node_vec = Set2Set(T=n_pass, n_hidden=n3)([x1_, x6])
    edge_vec = Set2Set(T=n_pass, n_hidden=n3)([x2_, x7])
    # concatenate atom, bond, and global
    final_vec = Concatenate(axis=-1)([node_vec, edge_vec, x3_])
    if dropout:
        final_vec = Dropout(dropout)(final_vec)
    # final dense layers
    final_vec = Dense(n2, activation=act)(final_vec)
    final_vec = Dense(n3, activation=act)(final_vec)
    out = Dense(n_target)(final_vec)
    model = Model(inputs=[x1, x2, x3, x4, x5, x6, x7], outputs=out)
    model.compile(Adam(lr), mse_scale)
    return model


def set2set_with_embedding_mp(n_connect,
                              n_global,
                              n_vocal=95,
                              embedding_dim=16,
                              n_blocks=3,
                              lr=1e-3,
                              n1=64,
                              n2=32,
                              n3=16,
                              n_pass=3,
                              n_target=1,
                              act=softplus2,
                              l2_coef=None,
                              is_classification=False):
    """
    construct a graph network model with only Z as atom features

    :param n_connect: (int) number of bond features
    :param n_global: (int) number of state features
    :param n_vocal: (int) number of vocabulary. Max Z number needs to be less than this.
        since we have max Z number 94 in materials project, here the default n_vocal = 95 (the last index is 94)
    :param embedding_dim: (int) embedding vector length
    :param n_blocks: (int) number of MEGNet block
    :param lr: (float) learning rate
    :param n1: (int) number of hidden units in layer 1 in MEGNet
    :param n2: (int) number of hidden units in layer 2 in MEGNet
    :param n3: (int) number of hidden units in layer 3 in MEGNet
    :param n_pass: (int) number of recurrent steps in Set2Set layer
    :param n_target: (int) number of output targets
    :param act: (object) activation function
    :param l2_coef: (float) l2 regularization rate
    :param is_classification: (bool) whether it is a classification problem
    :return: keras model object
    """
    int32 = 'int32'
    x1 = Input(shape=(None,), dtype=int32)
    x2 = Input(shape=(None, n_connect))
    x3 = Input(shape=(None, n_global))
    x4 = Input(shape=(None,), dtype=int32)
    x5 = Input(shape=(None,), dtype=int32)
    x6 = Input(shape=(None,), dtype=int32)
    x7 = Input(shape=(None,), dtype=int32)
    if l2_coef is not None:
        reg = l2(l2_coef)
    else:
        reg = None

    def ff(x, n_hiddens=[n1, n2]):
        out = x
        for i in n_hiddens:
            out = Dense(i, activation=act, kernel_regularizer=reg)(out)
        return out

    def one_block(a, b, c, has_ff=True):
        if has_ff:
            x1_ = ff(a)
            x2_ = ff(b)
            x3_ = ff(c)
        else:
            x1_ = a
            x2_ = b
            x3_ = c

        out = MEGNet([n1, n1, n2], [n1, n1, n2], [n1, n1, n2], pool_method='mean', activation=act,
                       kernel_regularizer=reg)(
            [x1_, x2_, x3_, x4, x5, x6, x7])

        return out[0], out[1], out[2]

    x1_ = Embedding(n_vocal, embedding_dim)(x1)
    x1_ = ff(x1_)
    x2_ = ff(x2)
    x3_ = ff(x3)
    for i in range(n_blocks):
        if i == 0:
            has_ff = False
        else:
            has_ff = True
        x1_1 = x1_
        x2_1 = x2_
        x3_1 = x3_
        x1_1, x2_1, x3_1 = one_block(x1_1, x2_1, x3_1, has_ff)
        x1_ = Add()([x1_, x1_1])
        x2_ = Add()([x2_, x2_1])
        x3_ = Add()([x3_, x3_1])
    node_vec = Set2Set(T=n_pass, n_hidden=n3)([x1_, x6])
    edge_vec = Set2Set(T=n_pass, n_hidden=n3)([x2_, x7])
    final_vec = Concatenate(axis=-1)([node_vec, edge_vec, x3_])
    final_vec = Dense(n2, activation=act)(final_vec)
    final_vec = Dense(n3, activation=act)(final_vec)
    if is_classification:
        final_act = 'sigmoid'
        loss = 'binary_crossentropy'
    else:
        final_act = None
        loss = mse_scale

    out = Dense(n_target, activation=final_act)(final_vec)
    model = Model(inputs=[x1, x2, x3, x4, x5, x6, x7], outputs=out)
    model.compile(Adam(lr), loss)
    return model
