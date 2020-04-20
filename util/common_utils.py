import matplotlib.pyplot as plot
import numpy as np
import torch as torch
import torch.optim as optim
from PIL import Image as Image
from PIL import ImageDraw as imDraw
from PIL import ImageFont as imFont
import os
import random
import torchvision
from math import log10, sqrt
from skimage.measure import compare_psnr



def get_mask(image, mask_type, max_ratio=0.50):
        return get_mask_with_noise(image, max_ratio)


def get_mask_with_noise(image, max_ratio):
    (h, w) = image.size

    text_font = imFont.load_default();

    img_mask_np = (np.random.random_sample(size=image_to_ndarray(image).shape) > max_ratio).astype(int)

    return ndarray_to_image(img_mask_np)


def image_to_ndarray(image):
    arr = np.array(image)
    if len(arr.shape) == 3:
        arr = arr.transpose(2, 0, 1)
    else:
        arr = arr[None, ...]

    return arr.astype(np.float32) / 255.


def ndarray_to_image(ndarray):
    array = np.clip(ndarray * 255, 0, 255).astype(np.uint8)

    if ndarray.shape[0] == 1:
        array = array[0]
    else:
        array = array.transpose(1, 2, 0)

    return Image.fromarray(array)
