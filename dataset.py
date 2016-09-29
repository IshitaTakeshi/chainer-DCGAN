import os
from os.path import join

import numpy as np
from scipy.misc import imresize
from skimage.io import imread
from tqdm import tqdm


def load(path, image_shape):
    image = imread(path)
    image = imresize(image, image_shape)
    image = (image - 128.) / 128.  # TODO make sure -1 <= image <= 1
    image = image.astype(np.float32).transpose(2, 0, 1)
    return image


def load_images(image_dir, image_shape):
    paths = [join(image_dir, f) for f in os.listdir(image_dir)[:200]]
    return np.array([load(path, image_shape) for path in tqdm(paths)])
