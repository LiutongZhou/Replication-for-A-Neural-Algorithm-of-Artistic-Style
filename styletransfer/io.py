import errno
import hashlib
import os

from PIL import Image


def check_path(path):
    """
    raises error if file does not exists

    :param path: str
    """
    if not os.path.exists(path):
        raise Exception(errno.ENOENT, "No such file", path)


def check_md5(file, blocksize=65536):
    """
    return the MD5 vaule of the local file

    :param file: path to file

    :type file: str

    :return:MD5

    :rtype: str
    """
    hasher = hashlib.md5()
    with open(file, 'rb') as afile:
        buf = afile.read(blocksize)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(blocksize)
    return hasher.hexdigest()


def load_image(file_path, verbose=0):
    """
    Read image from file, return a rgb image object of size (W,H,C)

    :param file_path: path to image

    :type file_path: string

    :param verbose: if 1, disp more info. defaults to 0

    :type verbose: int

    :return: im

    :rtype: Pillow Image object
    """
    check_path(file_path)
    im = Image.open(file_path).convert(mode="RGB")
    if verbose:
        print('image {} loaded'.format(file_path))
        print(im.format, im.size, im.mode)
    return im


def save_image(im, filepath):
    """
    Write image to filepath

    :param im: rgb image

    :type im: PIL.Image.Image

    :param filepath: file path to write

    :type filepath: str

    :return: img

    :rtype: Pillow image object

    """
    im.save(filepath)
    print('image file saved to ', filepath)


def maybe_make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
