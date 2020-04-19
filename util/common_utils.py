import matplotlib.pyplot as plot
import numpy as np
import torch as torch
import torch.optim as optim
from PIL import Image as Image
from PIL import ImageDraw as imDraw
from PIL import ImageFont as imFont
import os
import random
import lorem as lorem
import torchvision


def get_mask(image, mask_type, max_ratio=0.50):
    if (mask_type == 1):
        return get_mask_with_text(image)
    elif (mask_type == 2):
        return get_mask_with_noise(image, max_ratio)


def get_mask_with_text(image):
    (h, w) = image.size
    text_font = imFont.load_default();
    img_mask = Image.fromarray(np.array(image) * 0 + 255)
    overlay = imDraw.Draw(img_mask)

    for i in range(102):
        pos_x = random.randrange(start=0, stop=w)
        pos_y = random.randrange(start=0, stop=h)
        text = lorem.get_word(count=1)
        overlay.text(xy=(pos_x, pos_y), text=text, fill='rgb(0, 0, 0)', font=text_font)

    return img_mask


def get_mask_with_noise(image, max_ratio):
    (h, w) = image.size

    text_font = imFont.load_default();
    img_mask = Image.fromarray(np.array(image) * 0 + 255)

    img_mask_np = (np.random.random_sample(size=np.array(image).shape) > max_ratio).astype(int)

    return img_mask_np


## function directly used from source
def get_noise(input_depth, method, spatial_size, noise_type='u', var=1. / 10):
    shape = [1, input_depth, spatial_size[0], spatial_size[1]]
    net_input = torch.zeros(shape)

    fill_noise(net_input, noise_type)
    net_input *= var

    return net_input


## function directly used from source
def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False


## function directly used from source
def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.
    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:

        if opt == 'net':
            params += [x for x in net.parameters()]
        elif opt == 'down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'

    return params


def adam_optimize(parameters, closure, LR, num_iter):
    print('Optimizing with ADAM')
    optimizer = torch.optim.Adam(parameters, lr=LR)
    for j in range(num_iter):
        optimizer.zero_grad()
        closure()
        optimizer.step()


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
