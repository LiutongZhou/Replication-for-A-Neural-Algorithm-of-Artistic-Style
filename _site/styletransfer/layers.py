import tensorflow as tf


def conv2d(input_tensor, kernel, bias,layer_name, verbose=0):
    """
    conv2d returns a 2d convolution layer with full stride.

    :param input_tensor: A 4-D tensor

    :param kernel: A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]

    :param bias: biases of shape (num_kernels,), bias[i] is the ith bias corresponding to the ith conv kernel
    """
    conv = tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME', name=layer_name)
    if verbose:
        print('--{} | output_shape={} | kernel_shape={}'.format(layer_name,
                                                         conv.get_shape(),
                                                         kernel.get_shape()
                                                         )
              )
    return tf.nn.relu(conv+bias)


def pooling_2x2(input_tensor, layer_name, pooling_type='avg', verbose=0):
    """
    pooling_2x2 downsamples a feature map by 2X.

    :param input_tensor: A 4-D Tensor of shape [batch, height, width, channels]

    :param pooling_type: either 'avg' or 'max'
    """
    if pooling_type == 'avg':
        pooled = tf.nn.avg_pool(input_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=layer_name)
    elif pooling_type == 'max':
        pooled = tf.nn.max_pool(input_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=layer_name)
    if verbose:
        print('--{}   | shape={}'.format(layer_name, pooled.get_shape()
                                         )
              )
    return pooled
