import os

import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from six.moves.urllib.request import urlretrieve
from PIL import Image
import PIL

from .io import check_md5
from .layers import conv2d, pooling_2x2


class VGG19:
    """
    All data use the tensorflow default NHWC form
    
    """

    def __init__(self, weight_cache_path):
        """

        :param weight_cache_path: filepath to pretrained vgg19 model

        :type weight_cache_path: str
        """
        self.weight_cache_path = weight_cache_path
        self._vgg_layer_weights = None
        self.architecture = None
        self.input_gate = None

    def load_weights(self, path):
        """
        load pretrained vgg19 weights

        :param path: Full path to the pretrained vgg19 model

        :type path: str

        :return: the pretrained vgg19 model weights

        :rtype: np.ndarray
        """
        MD5 = '106118b7cf60435e6d8e04f6a6dc3657'
        URL = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'

        if not os.path.exists(path) or check_md5(path) != MD5:
            print('Downloading pretrained model weights to', path)
            urlretrieve(URL, path);

        assert check_md5(path) == MD5
        vgg19_weights = loadmat(path)
        self._vgg_layer_weights = vgg19_weights['layers'][0]
        return self._vgg_layer_weights

    def get_weights(self, i):
        """
        retrieve pretrained weights from vgg19.mat file

        :return: the retrived tensor

        :rtype: tf.tensor
        """
        weights = self._vgg_layer_weights[i][0][0][2][0][0]
        W = tf.constant(weights)
        return W

    def get_bias(self, i):
        """
        retrive pretrained biases from vgg19.mat file
        """
        bias = self._vgg_layer_weights[i][0][0][2][0][1]
        b = tf.constant(np.ravel(bias))
        return b

    def build(self, input_size, pooling_type='avg', verbose=0):
        """
        Build the vgg19 architecture

        It is a sequential model of multiple ((conv + relu)+[pooling]) layers

        :param input_size: Size of the input image. Follows (H,W,C) convention

        :type input_size: tuple

        :param pooling_type: either 'avg' or 'max'

        :param verbose: if 1, print detailed log

        :type verbose: int

        :return: the whole architecture layers referenced by a dict

        :rtype: dict
        """

        H, W, C = input_size
        if verbose: print('BUILDING VGG-19 Architecture')
        architecture = dict()

        if verbose: print('loading model weights...')
        _=self.load_weights(self.weight_cache_path)

        if verbose: print('constructing layers...')
        with tf.name_scope('Input'):
            # input is set to be a variable so it can be optimized over
            architecture['input'] = tf.Variable(np.zeros((1, H, W, C)), dtype=tf.float32, name='vgg_input')

        if verbose: print('Layer_Group_1')
        with tf.name_scope('Layer_Group_1'):
            with tf.name_scope('conv1_1'):
                architecture['conv1_1'] = conv2d(architecture['input'],
                                                 kernel=self.get_weights(0),
                                                 bias=self.get_bias(0),
                                                 layer_name='conv1_1',
                                                 verbose=verbose)

            with tf.name_scope('conv1_2'):
                architecture['conv1_2'] = conv2d(architecture['conv1_1'],
                                                 kernel=self.get_weights(2),
                                                 bias=self.get_bias(2),
                                                 layer_name='conv1_2',
                                                 verbose=verbose)

            with tf.name_scope('pooling1'):
                architecture['pool1'] = pooling_2x2(architecture['conv1_2'],
                                                    layer_name='pool1',
                                                    pooling_type=pooling_type,
                                                    verbose=verbose)

        if verbose: print('Layer_Group_2')
        with tf.name_scope('Layer_Group_2'):
            with tf.name_scope('conv2_1'):
                architecture['conv2_1'] = conv2d(architecture['pool1'],
                                                 kernel=self.get_weights(5),
                                                 bias=self.get_bias(5),
                                                 layer_name='conv2_1',
                                                 verbose=verbose)

            with tf.name_scope('conv2_2'):
                architecture['conv2_2'] = conv2d(architecture['conv2_1'],
                                                 kernel=self.get_weights(7),
                                                 bias=self.get_bias(7),
                                                 layer_name='conv2_2',
                                                 verbose=verbose)
            with tf.name_scope('pooling2'):
                architecture['pool2'] = pooling_2x2(architecture['conv2_2'],
                                                    layer_name='pool2',
                                                    pooling_type=pooling_type,
                                                    verbose=verbose)

        if verbose: print('Layer_Group_3')
        with tf.name_scope('Layer_Group_3'):
            with tf.name_scope('conv3_1'):
                architecture['conv3_1'] = conv2d(architecture['pool2'],
                                                 kernel=self.get_weights(10),
                                                 bias=self.get_bias(10),
                                                 layer_name='conv3_1',
                                                 verbose=verbose)

            with tf.name_scope('conv3_2'):
                architecture['conv3_2'] = conv2d(architecture['conv3_1'],
                                                 kernel=self.get_weights(12),
                                                 bias=self.get_bias(12),
                                                 layer_name='conv3_2',
                                                 verbose=verbose)

            with tf.name_scope('conv3_3'):
                architecture['conv3_3'] = conv2d(architecture['conv3_2'],
                                                 kernel=self.get_weights(14),
                                                 bias=self.get_bias(14),
                                                 layer_name='conv3_3',
                                                 verbose=verbose)

            with tf.name_scope('conv3_4'):
                architecture['conv3_4'] = conv2d(architecture['conv3_3'],
                                                 kernel=self.get_weights(16),
                                                 bias=self.get_bias(16),
                                                 layer_name='conv3_4',
                                                 verbose=verbose)

            with tf.name_scope('pooling3'):
                architecture['pool3'] = pooling_2x2(architecture['conv3_4'],
                                                    layer_name='pool3',
                                                    pooling_type=pooling_type,
                                                    verbose=verbose)

        if verbose: print('Layer_Group_4')
        with tf.name_scope('Layer_Group_4'):
            with tf.name_scope('conv4_1'):
                architecture['conv4_1'] = conv2d(architecture['pool3'],
                                                 kernel=self.get_weights(19),
                                                 bias=self.get_bias(19),
                                                 layer_name='conv4_1',
                                                 verbose=verbose)

            with tf.name_scope('conv4_2'):
                architecture['conv4_2'] = conv2d(architecture['conv4_1'],
                                                 kernel=self.get_weights(21),
                                                 bias=self.get_bias(21),
                                                 layer_name='conv4_2',
                                                 verbose=verbose)
            with tf.name_scope('conv4_3'):
                architecture['conv4_3'] = conv2d(architecture['conv4_2'],
                                                 kernel=self.get_weights(23),
                                                 bias=self.get_bias(23),
                                                 layer_name='conv4_3',
                                                 verbose=verbose)
            with tf.name_scope('conv4_4'):
                architecture['conv4_4'] = conv2d(architecture['conv4_3'],
                                                 kernel=self.get_weights(25),
                                                 bias=self.get_bias(25),
                                                 layer_name='conv4_4',
                                                 verbose=verbose)
            with tf.name_scope('pooling4'):
                architecture['pool4'] = pooling_2x2(architecture['conv4_4'],
                                                    layer_name='pool4',
                                                    pooling_type=pooling_type,
                                                    verbose=verbose)

        if verbose: print('Layer_Group_5')
        with tf.name_scope('Layer_Group_5'):
            with tf.name_scope('conv5_1'):
                architecture['conv5_1'] = conv2d(architecture['pool4'],
                                                 kernel=self.get_weights(28),
                                                 bias=self.get_bias(28),
                                                 layer_name='conv5_1',
                                                 verbose=verbose)
            with tf.name_scope('conv5_2'):
                architecture['conv5_2'] = conv2d(architecture['conv5_1'],
                                                 kernel=self.get_weights(30),
                                                 bias=self.get_bias(30),
                                                 layer_name='conv5_2',
                                                 verbose=verbose)
            with tf.name_scope('conv5_3'):
                architecture['conv5_3'] = conv2d(architecture['conv5_2'],
                                                 kernel=self.get_weights(32),
                                                 bias=self.get_bias(32),
                                                 layer_name='conv5_3',
                                                 verbose=verbose)
            with tf.name_scope('conv5_4'):
                architecture['conv5_4'] = conv2d(architecture['conv5_3'],
                                                 kernel=self.get_weights(34),
                                                 bias=self.get_bias(34),
                                                 layer_name='conv5_4',
                                                 verbose=verbose)
            with tf.name_scope('pooling5'):
                architecture['pool5'] = pooling_2x2(architecture['conv5_4'],
                                                    layer_name='pool5',
                                                    pooling_type=pooling_type,
                                                    verbose=verbose)

        self.architecture = architecture
        self.input_gate = architecture['input']

    def get_layer_output(self,input_data,ouput_layer):
        """
        feed the input data and retrieve the ouput tensor of output layer in the computational graph

        :param input_data: data to be feeded into the input gate

        :type input_data: np.ndarray

        :param ouput_layer: node name of the output layer

        :type ouput_layer: str

        :return: the output value of output_layer

        :rtype: np.ndarray

        """

        with tf.Session() as sess:
            sess.run(self.architecture['input'].assign(input_data))
            return sess.run(self.architecture[ouput_layer])


    @classmethod
    def preprocess(self, img):
        """
        Transform the image object to numpy array, and subtract a magic number for centering purpose.

        # note: the magic number is actually the normalization coefficient used by vgg19

        :param img: RGB image object of shape (W,H)

        :type img: PIL.Image.Image

        :return: RGB image array of shape (1,H,W,D) after subtracting the magic number

        :rtype: np.ndarray
        """
        img_array = np.array(img, dtype=np.float32)  # img_array.shape = (H,W,D)

        # shape (h, w, d) to (1, h, w, d)
        assert img_array.ndim == 3, "image data should have three channels"
        img_array = np.expand_dims(img_array, axis=0)
        img_array -= np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))  # subtract the magic number
        return img_array


    @classmethod
    def postprocess(self, img_data):
        """
        Reverse transformation of preprocessing

        :param img_data: RGB np.ndarray of shape (1,H,W,D) after subtracting the magic number

        :type img_data: np.ndarray

        :return: a Pillow image object of size (W,H)

        :rtype: PIL.Image.Image
        """
        img_data += np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
        # shape (1, h, w, d) to (h, w, d)
        img_data = img_data[0]
        img_data = np.clip(img_data, 0, 255).astype('uint8')
        assert img_data.ndim == 3, "image data should have three channels"
        im = Image.fromarray(img_data, "RGB")
        return im
