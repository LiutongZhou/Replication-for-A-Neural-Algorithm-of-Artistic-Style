from scipy.io import loadmat
from six.moves.urllib.request import urlretrieve
from styletransfer.io import check_md5
import os


class vgg19:
    def __init__(self):
        pass

    def load_weights(self,path):
        """
        :param path: Full path to the pretrained vgg19 model

        :type path: str

        :return: the pretrained vgg19 model weights

        :rtype: np.ndarray
        """
        MD5 = '106118b7cf60435e6d8e04f6a6dc3657'
        URL = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
        if not os.path.exists(path) or check_md5(path)!= MD5 :
            print('Downloading pretrained model weights to',path)
            urlretrieve(URL,path);

        assert check_md5(path) == MD5
        vgg19_weights = loadmat(path)
        return vgg19_weights['layers'][0]


    #todo
    def build_model(input_img):
        if args.verbose: print('\nBUILDING VGG-19 NETWORK')
        net = {}
        _, h, w, d = input_img.shape

        if args.verbose: print('loading model weights...')
        vgg_rawnet = scipy.io.loadmat(args.model_weights)
        vgg_layers = vgg_rawnet['layers'][0]
        if args.verbose: print('constructing layers...')

        net['input'] = tf.Variable(np.zeros((1, h, w, d), dtype=np.float32))

        if args.verbose: print('LAYER GROUP 1')
        net['conv1_1'] = conv_layer('conv1_1', net['input'], W=get_weights(vgg_layers, 0))
        net['relu1_1'] = relu_layer('relu1_1', net['conv1_1'], b=get_bias(vgg_layers, 0))

        net['conv1_2'] = conv_layer('conv1_2', net['relu1_1'], W=get_weights(vgg_layers, 2))
        net['relu1_2'] = relu_layer('relu1_2', net['conv1_2'], b=get_bias(vgg_layers, 2))

        net['pool1'] = pool_layer('pool1', net['relu1_2'])

        if args.verbose: print('LAYER GROUP 2')
        net['conv2_1'] = conv_layer('conv2_1', net['pool1'], W=get_weights(vgg_layers, 5))
        net['relu2_1'] = relu_layer('relu2_1', net['conv2_1'], b=get_bias(vgg_layers, 5))

        net['conv2_2'] = conv_layer('conv2_2', net['relu2_1'], W=get_weights(vgg_layers, 7))
        net['relu2_2'] = relu_layer('relu2_2', net['conv2_2'], b=get_bias(vgg_layers, 7))

        net['pool2'] = pool_layer('pool2', net['relu2_2'])

        if args.verbose: print('LAYER GROUP 3')
        net['conv3_1'] = conv_layer('conv3_1', net['pool2'], W=get_weights(vgg_layers, 10))
        net['relu3_1'] = relu_layer('relu3_1', net['conv3_1'], b=get_bias(vgg_layers, 10))

        net['conv3_2'] = conv_layer('conv3_2', net['relu3_1'], W=get_weights(vgg_layers, 12))
        net['relu3_2'] = relu_layer('relu3_2', net['conv3_2'], b=get_bias(vgg_layers, 12))

        net['conv3_3'] = conv_layer('conv3_3', net['relu3_2'], W=get_weights(vgg_layers, 14))
        net['relu3_3'] = relu_layer('relu3_3', net['conv3_3'], b=get_bias(vgg_layers, 14))

        net['conv3_4'] = conv_layer('conv3_4', net['relu3_3'], W=get_weights(vgg_layers, 16))
        net['relu3_4'] = relu_layer('relu3_4', net['conv3_4'], b=get_bias(vgg_layers, 16))

        net['pool3'] = pool_layer('pool3', net['relu3_4'])

        if args.verbose: print('LAYER GROUP 4')
        net['conv4_1'] = conv_layer('conv4_1', net['pool3'], W=get_weights(vgg_layers, 19))
        net['relu4_1'] = relu_layer('relu4_1', net['conv4_1'], b=get_bias(vgg_layers, 19))

        net['conv4_2'] = conv_layer('conv4_2', net['relu4_1'], W=get_weights(vgg_layers, 21))
        net['relu4_2'] = relu_layer('relu4_2', net['conv4_2'], b=get_bias(vgg_layers, 21))

        net['conv4_3'] = conv_layer('conv4_3', net['relu4_2'], W=get_weights(vgg_layers, 23))
        net['relu4_3'] = relu_layer('relu4_3', net['conv4_3'], b=get_bias(vgg_layers, 23))

        net['conv4_4'] = conv_layer('conv4_4', net['relu4_3'], W=get_weights(vgg_layers, 25))
        net['relu4_4'] = relu_layer('relu4_4', net['conv4_4'], b=get_bias(vgg_layers, 25))

        net['pool4'] = pool_layer('pool4', net['relu4_4'])

        if args.verbose: print('LAYER GROUP 5')
        net['conv5_1'] = conv_layer('conv5_1', net['pool4'], W=get_weights(vgg_layers, 28))
        net['relu5_1'] = relu_layer('relu5_1', net['conv5_1'], b=get_bias(vgg_layers, 28))

        net['conv5_2'] = conv_layer('conv5_2', net['relu5_1'], W=get_weights(vgg_layers, 30))
        net['relu5_2'] = relu_layer('relu5_2', net['conv5_2'], b=get_bias(vgg_layers, 30))

        net['conv5_3'] = conv_layer('conv5_3', net['relu5_2'], W=get_weights(vgg_layers, 32))
        net['relu5_3'] = relu_layer('relu5_3', net['conv5_3'], b=get_bias(vgg_layers, 32))

        net['conv5_4'] = conv_layer('conv5_4', net['relu5_3'], W=get_weights(vgg_layers, 34))
        net['relu5_4'] = relu_layer('relu5_4', net['conv5_4'], b=get_bias(vgg_layers, 34))

        net['pool5'] = pool_layer('pool5', net['relu5_4'])

        return net
