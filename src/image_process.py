import os
from PIL import Image, ImageOps
import numpy as np
import scipy.misc


def get_resized_image(img_path, height, width):
	# resize the image to required size
    image = Image.open(img_path)
    image = ImageOps.fit(image, (width, height), Image.ANTIALIAS)
    image = np.asarray(image, np.float32)
    return np.expand_dims(image, 0)

def generate_noise_image(content_image, height, width, noise_ratio=0.6):
	# add noise to image
    noise_image = np.random.uniform(-20, 20, (1, height, width, 3)).astype(np.float32)
    return noise_image * noise_ratio + content_image * (1 - noise_ratio)

def save_image(path, image):
	# save image
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)