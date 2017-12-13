import numpy as np
import tensorflow as tf
import scipy.io


def _cov2d_layer(vgg_layers, prev_layer, layer, layer_name):
    """ 
    Generate the Conv2D layer with RELU
    """
    with tf.variable_scope(layer_name) as scope:
        # get weight from pretrained VGG
        W = tf.constant(vgg_layers[layer][0][0][2][0][0], name='weights')

        # get bias from pretrained VGG
        b = vgg_layers[layer][0][0][2][0][1]
        c = b.reshape(b.size)
        b = tf.constant(c, name='bias')

        # conv2d with bias added
        conv2d = tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b
    return tf.nn.relu(conv2d)

def _avg_pool(prev_layer):
    """ 
    Return the average pooling layer. 
    The paper suggests that average pooling actually works better than max pooling.
    """
    return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                          padding='SAME', name='avg_pool_')

def build_vgg(path, input_image):
    """ 
    Load VGG into a TensorFlow model.
    """
    vgg_pretrained = scipy.io.loadmat(path)
    vgg_layers = vgg_pretrained['layers'][0]

    model = {} 
    model['conv1_1']  = _cov2d_layer(vgg_layers, input_image, 0, 'conv1_1')
    model['conv1_2']  = _cov2d_layer(vgg_layers, model['conv1_1'], 2, 'conv1_2')
    model['avgpool1'] = _avg_pool(model['conv1_2'])
    model['conv2_1']  = _cov2d_layer(vgg_layers, model['avgpool1'], 5, 'conv2_1')
    model['conv2_2']  = _cov2d_layer(vgg_layers, model['conv2_1'], 7, 'conv2_2')
    model['avgpool2'] = _avg_pool(model['conv2_2'])
    model['conv3_1']  = _cov2d_layer(vgg_layers, model['avgpool2'], 10, 'conv3_1')
    model['conv3_2']  = _cov2d_layer(vgg_layers, model['conv3_1'], 12, 'conv3_2')
    model['conv3_3']  = _cov2d_layer(vgg_layers, model['conv3_2'], 14, 'conv3_3')
    model['conv3_4']  = _cov2d_layer(vgg_layers, model['conv3_3'], 16, 'conv3_4')
    model['avgpool3'] = _avg_pool(model['conv3_4'])
    model['conv4_1']  = _cov2d_layer(vgg_layers, model['avgpool3'], 19, 'conv4_1')
    model['conv4_2']  = _cov2d_layer(vgg_layers, model['conv4_1'], 21, 'conv4_2')
    model['conv4_3']  = _cov2d_layer(vgg_layers, model['conv4_2'], 23, 'conv4_3')
    model['conv4_4']  = _cov2d_layer(vgg_layers, model['conv4_3'], 25, 'conv4_4')
    model['avgpool4'] = _avg_pool(model['conv4_4'])
    model['conv5_1']  = _cov2d_layer(vgg_layers, model['avgpool4'], 28, 'conv5_1')
    model['conv5_2']  = _cov2d_layer(vgg_layers, model['conv5_1'], 30, 'conv5_2')
    model['conv5_3']  = _cov2d_layer(vgg_layers, model['conv5_2'], 32, 'conv5_3')
    model['conv5_4']  = _cov2d_layer(vgg_layers, model['conv5_3'], 34, 'conv5_4')
    model['avgpool5'] = _avg_pool(model['conv5_4'])
    
    return model