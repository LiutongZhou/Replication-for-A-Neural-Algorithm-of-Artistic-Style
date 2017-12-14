import argparse

import os
import time

import numpy as np
import scipy.io
import tensorflow as tf
from styletransfer.io import  check_path, load_image



class artist:
    """
    Arguments:

    Methods:
        get_content: load the content image, preprocess it, return a np array for use

    Attributes:

        content_image: Pillow image object, only available after explicitly calling fit or get_content

        content_image_shape: (H,W,D) tuple, only available after calling get_content

        style_images: a list of image objects, only only available after explicitly calling fit or get_styles


    """
    def __init__(self,
                 content_dir= './data/content/',
                 style_dir = './data/style/',
                 init_type= 'content',
                 noise_ratio = 1.,
                 max_szie=512,
                 **kwargs):
        """


        :param content_dir: path to directory containing content images

        :type content_dir: str

        :param style_dir: path to directory containing style images

        :type style_dir: str

        :param init_type: The inital target image. either 'content', 'tyle', or 'random'.

        :param noise_ratio: 0. ~ 1. denoting the percentage to which the initial target image
                            favors white noise, default to 1 (100% white noise). Only effective when use
                            'random' init_type

        :param max_szie: Maximum width or height of the input images

        :type max_szie: int


        """
        self.content_dir = content_dir
        self.max_size = max_szie
        self.style_dir = style_dir
        self.init_type = init_type
        self.noise_ratio = noise_ratio
        self.content_image = None
        self.style_images = None
        self.pretrained_vgg19_path = kwargs.get('pretrained_vgg19_path',
                                                './data/imagenet-vgg-verydeep-19')




    def get_content(self, content_img):
        """
        load the content image, preprocess it, return a np array for use

        :param content_img: filename of the content image

        :type content_img: str

        :return: the loaded content data after preprocessing, shape = (1, H, W, D)

        :rtype: np.ndarray
        """
        path = os.path.join(self.content_dir, content_img)
        check_path(path)
        img = load_image(path)  # RGB pillow image object

        h, w = img.size
        mx = self.max_size

        # resize if > max size
        if h > w and h > mx:
            new_h = mx
            new_w = new_h / h * w
            new_size = (int(new_w), int(new_h))
            img = img.resize(new_size)

        if w > mx:  # if w>h and w>mx
            new_w = mx
            new_h = new_w / w * h
            new_size = (int(new_w), int(new_h))
            img = img.resize(new_size)

        self.content_img = img

        img_array = self._preprocess(img)

        _,ch,cw,cd = img_array.shape
        self.content_image_shape = (ch,cw,cd)
        self.__content_img_array = img_array

        return img_array

    def get_styles(self,style_imgs= 'All'):
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

        self.styles = styles

        # load style images
        assert self.content_img is not None, "content image not loaded yet, get_content first"
        H, W, _ = self.content_image_shape

        style_paths = map(lambda x: os.path.join(self.style_dir, x),self.styles)
        style_imgs = []
        style_image_arrays = []
        for path in style_paths:
            check_path(path)
            img = load_image(path)
            img = img.resize((W,H))
            style_imgs.append(img)
            img_array = self._preprocess(img)
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
        noise_img = np.random.uniform(-20., 20., img_data.shape).astype(np.float32)
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


    @classmethod
    def _preprocess(self, img):
        """
        Transform the image object to numpy array, and subtract a magic number for centering purpose.

        :param img: RGB image object of shape (W,H)

        :type img: PIL.Image.Image

        :return: RGB image array of shape (1,H,W,D) after subtracting the magic number

        :rtype: np.ndarray
        """
        img_array = np.array(img, dtype=np.int16)  # img_array.shape = (H,W,D)

        # shape (h, w, d) to (1, h, w, d)
        assert img_array.ndim == 3, "image data should have three channels"
        img_array = np.expand_dims(img_array, axis=0)
        img_array -= np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))  # subtract the magic number
        return img_array

    def fit(self,content,style = None):
        """
        learn to combine content and style

        :param content: content image name

        :type content: str

        :param style: style image name or a list of style image names. Automatically use all styles if not set

        :type style: str  or a list of strs

        """

        content_data = self.get_content(content)

        if style:
            style_data = self.get_styles(style_imgs=style)
        else:
            style_data = self.get_styles()

        with tf.Graph().as_default():
            print('='*30 + ' Fitting ' + '='*30 )
            tick = time.time()

            init_target_data = self._init_target_image()
            optimize(content_data, style_data, init_target_data)

            tock = time.time()
            print('Fitting done. Wall time: {}'.format(tock - tick))


    def draw(self):
        pass

