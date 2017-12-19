import os
import time

import numpy as np
import tensorflow as tf

from .io import check_path, load_image, save_image, maybe_make_directory
from .optimizer import Adam, LBFGS
from .vgg19 import VGG19
from PIL.Image import HAMMING


def _content_layer_loss(content_feature, target_image_feature, content_loss_function):
    """
    Return the content loss between image feature and content feature

    a feature is the extracted output from a vgg19 layer.

    content loss is defined as a weighted squared error

    :param content_feature: the extracted output tensor from a vgg19 layer, on feeding content data
                            as input

    :type content_feature: tf.Tensor

    :param target_image_feature: the extracted output tensor from a vgg19 layer, on feeding target
                                image data as input

    :type target_image_feature: tf.Tensor

    :return: content loss for the extracted content feature

    :rtype: tf.Tensor
    """
    _, h, w, d = content_feature.get_shape()
    M = h.value * w.value
    N = d.value
    if content_loss_function == 1:
        K = 1. / (2. * N ** 0.5 * M ** 0.5)
    elif content_loss_function == 2:
        K = 1. / (N * M)
    elif content_loss_function == 3:
        K = 1. / 2.
    loss = K * tf.reduce_sum((content_feature - target_image_feature) ** 2)
    return loss


def _gram_matrix(x):
    """
    calculate X'X

    :param x: input matrix X

    :type: tf.Tensor

    :return: X'X

    :rtype: tf.Tensor
    """
    return tf.matmul(tf.transpose(x), x)


def _style_layer_loss(style_feature, target_image_style_feature):
    """
    Return the style loss between target image style feature and style image feature

    a style feature represented by a conv2 layer is the correlation matrix A such that A[i,j] is the

    inner product between the ith feature map and jth feature map

    style loss is defined as a weighted squared error

    :param style_feature: the extracted output tensor from a vgg19 layer, on feeding style img data
                            as input

    :type style_feature: tf.Tensor

    :param target_image_style_feature: the extracted output tensor from a vgg19 layer, on feeding target
                                image data as input

    :type target_image_style_feature: tf.Tensor

    :return: style loss for the extracted style feature

    :rtype: tf.Tensor
    """

    _, h, w, d = style_feature.get_shape()
    M = h.value * w.value
    N = d.value  # number of filters at the conv2 layer

    # flatten every feature map corresponding to the ith filter to be a row vector in the ith row
    A = tf.reshape(style_feature, (M, N))
    # calculate the correlation matrix so that A[i,j] is correlation between the ith feature map and jth feature map
    A = _gram_matrix(A)

    # same for content feature map
    G = tf.reshape(target_image_style_feature, (M, N))
    G = _gram_matrix(G)

    loss = 1 / 4 * 1 / (N ** 2 * M ** 2) * tf.reduce_sum(tf.pow((G - A), 2))
    return loss


class Artist:
    """
        Methods:
        get_content: load the content image, preprocess it, return a np array for use

        Attributes:

        content_image: Pillow image object, only available after explicitly calling fit or get_content

        content_image_shape: (H,W,D) tuple, only available after calling get_content

        style_images: a list of image objects, only only available after explicitly calling fit or get_styles

    """

    def __init__(self,
                 content_dir='./data/content/',
                 style_dir='./data/style/',
                 output_dir = './data/output',
                 init_type='content',
                 noise_ratio=1.,
                 max_szie=512,
                 content_weight=5e0,
                 style_weight=1e4,
                 **kwargs):
        """


        :param content_dir: path to directory containing content images

        :type content_dir: str

        :param style_dir: path to directory containing style images

        :type style_dir: str

        :param output_dir: path to output directory

        :type output_dir: str

        :param init_type: The inital target image. either 'content', 'tyle', or 'random'.

        :param noise_ratio: 0. ~ 1. denoting the percentage to which the initial target image
                            favors white noise, default to 1 (100% white noise). Only effective when use
                            'random' init_type

        :param max_szie: Maximum width or height of the input images

        :type max_szie: int

        :param content_weight: Weight for the content loss function. This is referred to as alpha in the paper

        :type content_weight: float

        :param style_weight: Weight for the style loss function. This is referred to as betas in the paper

        :type style_weight: float

        :param content_layers: VGG19 layers used for the extracting features of content image, defaults to ['conv4_2']

        :type content_layers: list

        :param content_layer_weights: weights of each content layer to loss, defaults to [1.0]

        :param content_layer_weights: list

        :param style_layers: VGG19 layers used for the extracting features of style image,defaults to
                             ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

        :type style_layers: list

        :param style_layer_weights: defaults to uniform distribution

        :type style_layer_weights: list

        :param content_loss_function: if 1, use 1/(2 * N ** 0.5 * M ** 0.5) as coefficient when calculating
                                      content loss; if 2, use 1/(M*N), if 3, use 1/2

        :type content_loss_function: int

        """
        self.content_dir = content_dir
        self.max_size = max_szie
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.style_dir = style_dir
        self.output_dir = output_dir
        self.init_type = init_type
        self.noise_ratio = noise_ratio
        self.content_image = None  # Only available after calling get_content
        self.content_image_shape = None
        self.style_images = None

        self.content_layers = kwargs.get('content_layers', ['conv4_2'])
        self.style_layers = kwargs.get('style_layers', ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'])
        self.content_layer_weights = kwargs.get('content_layer_weights', [1/len(self.content_layers)]*len(self.content_layers))

        self.style_layer_weights = kwargs.get('style_layer_weights',
                                              [1 / len(self.style_layers)] * len(self.style_layers))

        self.pretrained_vgg19_path = kwargs.get('pretrained_vgg19_path', './data/imagenet-vgg-verydeep-19.mat')
        self.content_loss_function = kwargs.get('content_loss_function', 1)

    @property
    def style_layer_weights(self):
        return self.__style_layer_weights

    @style_layer_weights.setter
    def style_layer_weights(self, weights):
        denom = sum(weights)
        if denom > 0:
            self.__style_layer_weights = [float(i) / denom for i in weights]  # normalize to 0~1
        else:
            self.__style_layer_weights = [0.] * len(weights)

    @property
    def style_image_weights(self):
        return self.__style_image_weights

    @style_image_weights.setter
    def style_image_weights(self, weights):
        denom = sum(weights)
        if denom > 0:
            self.__style_image_weights = [float(i) / denom for i in weights]  # normalize to 0~1
        else:
            self.__style_image_weights = [0.] * len(weights)

    def get_content(self, content_img):
        """
        load the content image, preprocess it, return a np array for use

        :param content_img: filename of the content image

        :type content_img: str

        :return: the loaded content data after preprocessing, shape = (1, H, W, C)

        :rtype: np.ndarray
        """
        path = os.path.join(self.content_dir, content_img)
        check_path(path)
        img = load_image(path)  # RGB pillow image object

        w, h = img.size
        mx = self.max_size

        # resize if > max size
        if h > w and h > mx:
            new_h = mx
            new_w = new_h / h * w
            new_size = (int(new_w), int(new_h))
            img = img.resize(new_size,HAMMING)

        if w > mx:  # if w>h and w>mx
            new_w = mx
            new_h = new_w / w * h
            new_size = (int(new_w), int(new_h))
            img = img.resize(new_size,HAMMING)

        self.content_img = img

        img_array = VGG19.preprocess(img)

        _, ch, cw, cd = img_array.shape
        self.content_image_shape = (ch, cw, cd)
        self.__content_img_array = img_array

        return img_array

    def get_styles(self, style_imgs='All'):
        """
        load the style images, preprocess them, return a a list of np arrays for use

        :param style_imgs: filename of the style images

        :type style_imgs: str or a list of strings, if a list of trings, multiple styles are used

        :return: the loaded styles data after preprocessing, a list of np.ndarray of shape = (1, H, W, D)

        :rtype: list
        """

        if style_imgs == 'All':
            styles = [f for f in os.listdir(self.style_dir)
                      if os.path.isfile(os.path.join(self.style_dir, f))]
        elif isinstance(style_imgs, str):
            styles = [style_imgs]
        else:
            styles = style_imgs

        self.styles = styles

        # load style images
        assert self.content_img is not None, "content image not loaded yet, get_content first"
        H, W, _ = self.content_image_shape

        style_paths = map(lambda x: os.path.join(self.style_dir, x), self.styles)
        style_imgs = []
        style_image_arrays = []
        for path in style_paths:
            check_path(path)
            img = load_image(path)
            img = img.resize((W, H))
            style_imgs.append(img)
            img_array = VGG19.preprocess(img)
            style_image_arrays.append(img_array)

        self.style_images = style_imgs
        self.__style_image_arrays = style_image_arrays
        return style_image_arrays

    @staticmethod
    def get_white_noise_image(img_data, noise_ratio=1):
        """
        Generates a white noise image. This is mainly for use of initilizing the target image.

        :param img_data: for which image data we generate the white_noise_image

        :type img_data: np.ndarray

        :param noise_ratio: 0. ~ 1. denoting the percentage to which the generated image favors white noise,
                            default to 1 (100% white noise).

        :type noise_ratio: float

        :return: the white noise imgage data for use

        :rtype: np.ndarray
        """
        noise_img = np.random.uniform(-10., 10., img_data.shape).astype(np.float32)
        img_data = noise_ratio * noise_img + (1. - noise_ratio) * img_data
        return img_data

    def _init_target_image(self):
        init_type = self.init_type

        if init_type == 'content':
            return self.__content_img_array
        elif init_type == 'style':
            return self.__style_image_arrays[0]
        elif init_type == 'random':
            return self.get_white_noise_image(self.__content_img_array, noise_ratio=self.noise_ratio)

    def get_loss(self, vgg19):
        """
        get the total loss to be minimized

        :return:
        """
        with tf.Session() as sess:

            # get content loss
            content_loss = 0.
            # for every layer in the content_layers that are specified to output feature representations do:
            for weight, layer in zip(self.content_layer_weights, self.content_layers):
                target_image_feature = vgg19.architecture[layer]  # a variable
                content_feature = vgg19.get_layer_output(self.__content_img_array, layer)  # extract content_feature
                content_feature = tf.convert_to_tensor(content_feature)  # a constant

                content_loss += weight * _content_layer_loss(content_feature,
                                                             target_image_feature,
                                                             self.content_loss_function)
            content_loss /= float(len(self.content_layers))  # normailization
            tf.summary.scalar('content_loss', content_loss)

            # get style loss
            style_loss = 0.
            for img_weight, img_data in zip(self.style_image_weights, self.__style_image_arrays):
                loss = 0.
                for weight, layer in zip(self.style_layer_weights, self.style_layers):
                    target_image_feature = vgg19.architecture[layer]
                    style_feature = vgg19.get_layer_output(img_data, layer)
                    style_feature = tf.convert_to_tensor(style_feature)
                    loss += weight * _style_layer_loss(style_feature, target_image_feature)
                loss /= len(self.style_layers)
                style_loss += (loss * img_weight)
            style_loss /= len(self.__style_image_arrays)
            tf.summary.scalar('style_loss', style_loss)

            noise = tf.image.total_variation(vgg19.architecture['input'])

            total_loss = self.content_weight * content_loss + self.style_weight * style_loss + 1e-3 * noise
            tf.summary.scalar('total_loss', total_loss)
            return total_loss

    def fit_transform(self, content, style=None, style_image_weights=None,
                      pooling_type='avg',
                      optimizer='adam',
                      max_iter=1000,
                      verbose=0):
        """
        learn to combine content and style

        :param content: content image name

        :type content: str

        :param style: style image name or a list of style image names. Automatically use all styles if not set

        :type style: str  or a list of strs

        :param style_image_weights: if None, use uniform weights among multiple style images

        :type style_image_weights: list

        :param pooling_type: either 'avg' or 'max'

        :param optimizer: either 'adam' or 'l-bfgs'

        :param max_iter: max number of iterations for updating the image in the optimization process

        :param verbose: 0 or 1, if 1, print more details

        """

        # get content data
        content_data = self.get_content(content)

        # get style data
        if style:
            style_data = self.get_styles(style_imgs=style)
        else:
            style_data = self.get_styles()

        # get style image weights
        if style_image_weights is None:
            self.style_image_weights = [1 / len(style_data)] *len(style_data)
        else:
            self.style_image_weights = style_image_weights
        assert isinstance(self.style_image_weights, list), "style_image_weights must be a list"

        with tf.Graph().as_default(),  tf.Session().as_default() as sess:
            print('=' * 30 + ' Fitting ' + '=' * 30)
            tick = time.time()

            vgg19 = VGG19(self.pretrained_vgg19_path)
            vgg19.build(input_size=self.content_image_shape, pooling_type=pooling_type, verbose=verbose)

            total_loss = self.get_loss(vgg19)

            # init image
            init_target_data = self._init_target_image()
            sess.run(vgg19.architecture['input'].assign(init_target_data))

            tf.summary.image('image_to_generate', vgg19.architecture['input'], 5)
            summary_writer = tf.summary.FileWriter('./log', sess.graph) # Todo
            merged = tf.summary.merge_all() # Todo:  Add Tensorboard Visualization

            # set optimizer
            if optimizer.lower() == 'adam':
                optimizer = Adam(learning_rate=0.1)
            elif optimizer.lower() == 'l-bfgs':
                optimizer = LBFGS()

            # Train
            optimizer.train(total_loss, max_step=max_iter, verbose=verbose)
            self._generated_image_data = sess.run(vgg19.architecture['input'])
            summary_writer.close()

            tock = time.time()
            print('Fitting done. Wall time: {}'.format(tock - tick))
            self.generated_image = VGG19.postprocess(self._generated_image_data)
            return  self.generated_image

    def draw(self,name):
        maybe_make_directory(self.output_dir)
        self.generated_image.show()
        output_filepath =os.path.join(self.output_dir,name+'.png')
        save_image(self.generated_image,output_filepath)
        return self.generated_image
