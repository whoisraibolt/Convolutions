#!/usr/bin/python
# -*- coding: utf-8 -*-

# Imports
from PIL import Image
from scipy import signal as sg
import numpy as np

# Transform the image into an array (matrix)
def array_from_image(f_name):
    np_image = np.asarray(Image.open(f_name), dtype=np.float32)

    return np_image

# Normalize the data
def normalize(f_name):
    normalize = 255. * np.absolute(f_name) / np.max(f_name)

    return normalize

# Save array as image
def save_as_image(array, f_name):
    Image.fromarray(array.round().astype(np.uint8)).save(f_name)

# Opens the image and receives the array
img = array_from_image('Images/gate.png')

# Kernels
kernels = [[[0,  1, 0,], [0, -1, 0,], [0,  0, 0,],],
           [[0,  0, 1,], [0, -1, 0,], [0,  0, 0,],],
           [[0,  0, 0,], [0, -1, 1,], [0,  0, 0,],],
           [[0,  0, 0,], [0, -1, 0,], [0,  0, 1,],],
           [[0,  0, 0,], [0, -1, 0,], [0,  1, 0,],],
           [[0,  0, 0,], [0, -1, 0,], [1,  0, 0,],],
           [[0,  0, 0,], [1, -1, 0,], [0,  0, 0,],],
           [[1,  0, 0,], [0, -1, 0,], [0,  0, 0,],]]

# Saves the resulting image from the original image with each kernels
for i in range(0, len(kernels)):    
    save_as_image(normalize(sg.convolve(img, kernels[i])), 'Images/conv-'+str(i)+'.png')