import tensorflow as tf


def _repeat(x, n, axis=1):
    """
    Given an tensor x (N*M*K), repeat the middle axis (axis=1) according to repetition indicator n (M, )
    for example, if M = 3, axis=1, and n = Tensor([3, 1, 2]),
    and the final tensor would have the shape (N*6*3) with the first one in M repeated 3 times,
    second 1 time and third 2 times.

     Args:
        x: (3d Tensor) tensor to be augmented
        n: (1d Tensor) number of repetition for each row
        axis: (int) axis for repetition

    Returns:
        (3d Tensor) tensor after repetition
    """
    # get maximum repeat length in x
    assert len(n.shape) == 1
    maxlen = tf.reduce_max(n)
    x_shape = tf.shape(x)
    x_dim = len(x.shape)
    # get the length of x
    xlen = tf.shape(n)[0]
    # create a range with the length of x
    shape = [1] * (x_dim + 1)
    shape[axis + 1] = maxlen
    # tile it to the maximum repeat length, it should be of shape [xlen, maxlen] now
    x_tiled = tf.tile(tf.expand_dims(x, axis + 1), tf.stack(shape))

    new_shape = tf.unstack(x_shape)
    new_shape[axis] = -1
    new_shape[-1] = x.shape[-1]
    x_tiled = tf.reshape(x_tiled, new_shape)
    # create a sequence mask using x
    # this will create a boolean matrix of shape [xlen, maxlen]
    # where result[i,j] is true if j < x[i].
    mask = tf.sequence_mask(n, maxlen)
    mask = tf.reshape(mask, (-1,))
    # mask the elements based on the sequence mask
    return tf.boolean_mask(x_tiled, mask, axis=axis)


def repeat_with_index(x, index, axis=1):
    """
    Given an tensor x (N*M*K), repeat the middle axis (axis=1) according to the index tensor index (G, )
    for example, if axis=1 and n = Tensor([0, 0, 0, 1, 2, 2]) then M = 3 (3 unique values),
    and the final tensor would have the shape (N*6*3) with the first one in M repeated 3 times,
    second 1 time and third 2 times.

     Args:
        x: (3d Tensor) tensor to be augmented
        index: (1d Tensor) repetition tensor
        axis: (int) axis for repetition
    Returns:
        (3d Tensor) tensor after repetition
    """
    index = tf.reshape(index, (-1,))
    _, _, n = tf.unique_with_counts(index)
    return _repeat(x, n, axis)
