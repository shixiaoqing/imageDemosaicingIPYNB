# -*- coding: utf-8 -*-
"""

# Commented out IPython magic to ensure Python compatibility.

import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy
import skimage
from scipy import ndimage
from scipy import signal

# %matplotlib inline

"""### Reading the Mosaic Image"""

IMG_DIR = 'images/'
IMG_NAME = 'crayons.bmp'

def read_image(IMG_NAME):
    img = mpimg.imread(IMG_DIR + IMG_NAME)
    return img

mosaic_img = read_image(IMG_NAME)
plt.imshow(mosaic_img, cmap="gray")

"""### Linear Interpolation"""

def get_helper_arrays(mosaic_img):
  shape = np.shape(mosaic_img)
  mosaic_arr = skimage.img_as_float(mosaic_img)
  red_arr, blue_arr, green_arr = [np.zeros((shape[0], shape[1])) for _ in range(3)]

  # Gets values for each channel according to Bayer image
  green_arr[0::2, 1::2] = mosaic_arr[0::2, 1::2]
  green_arr[1::2, 0::2] = mosaic_arr[1::2, 0::2]

  red_arr[0::2, 0::2] = mosaic_arr[0::2, 0::2]

  blue_arr[1::2, 1::2] = mosaic_arr[1::2, 1::2]

  # Kernels
  rb_kernel = np.array([[1,2,1],
                        [2,4,2],
                        [1,2,1]])

  green_kernel = np.array([[0,1,0],
                           [1,4,1],
                           [0,1,0]])

  return red_arr, blue_arr, green_arr, rb_kernel, green_kernel

def get_solution_image(mosaic_img):
    mosaic_shape = np.shape(mosaic_img)
    soln_image = np.zeros((mosaic_shape[0], mosaic_shape[1], 3))

    red_arr, blue_arr, green_arr, rb_kernel, green_kernel = get_helper_arrays(mosaic_img)

    red_arr = ndimage.convolve(red_arr, rb_kernel/4)
    green_arr = ndimage.convolve(green_arr, green_kernel/4)
    blue_arr = ndimage.convolve(blue_arr,rb_kernel/4)

    soln_image = np.dstack((red_arr, green_arr, blue_arr))
    return soln_image

def compute_errors(soln_image, original_image):
    soln_img = np.array(soln_image)
    original_image = skimage.img_as_float(original_image)
    squared_diff = (original_image + soln_img) * (original_image - soln_img)

    # Max pixel error
    max_err = np.max(squared_diff)

    # Average error per pixel
    pp_err = np.average(squared_diff)

    return pp_err, max_err

"""We provide you with 3 images to test if your solution works. Once it works, you should generate the solution for test image provided to you."""

mosaic_img = read_image('crayons.bmp')
soln_image = get_solution_image(mosaic_img)
original_image = read_image('crayons.jpg')
plt.imshow(soln_image)

pp_err, max_err = compute_errors(soln_image, original_image)
print("The average per-pixel error for crayons is: "+str(pp_err))
print("The maximum per-pixel error for crayons is: "+str(max_err))

mosaic_img = read_image('iceberg.bmp')
soln_image = get_solution_image(mosaic_img)
original_image = read_image('iceberg.jpg')
plt.imshow(soln_image)

pp_err, max_err = compute_errors(soln_image, original_image)
print("The average per-pixel error for iceberg is: "+str(pp_err))
print("The maximum per-pixel error for iceberg is: "+str(max_err))

mosaic_img = read_image('tony.bmp')
soln_image = get_solution_image(mosaic_img)
original_image = read_image('tony.jpg')
plt.imshow(soln_image)

pp_err, max_err = compute_errors(soln_image, original_image)
print("The average per-pixel error for tony is: "+str(pp_err))
print("The maximum per-pixel error for tony is: "+str(max_err))

mosaic_img = read_image('hope.bmp')
soln_image = get_solution_image(mosaic_img)
plt.imshow(soln_image)

def get_freeman_solution_image(mosaic_img):
    mosaic_shape = np.shape(mosaic_img)

    black = np.zeros((mosaic_shape[0], mosaic_shape[1]))

    red_arr, blue_arr, green_arr, rb_kernel, green_kernel = get_helper_arrays(mosaic_img)

    green_arr = ndimage.convolve(green_arr, green_kernel/4)
    red_arr = ndimage.convolve(red_arr, rb_kernel/4) - green_arr
    blue_arr = ndimage.convolve(blue_arr,rb_kernel/4) - green_arr

    red_arr = signal.medfilt2d(red_arr) + green_arr
    blue_arr = signal.medfilt2d(blue_arr) + green_arr

    freeman_soln_image = np.dstack((red_arr, green_arr, blue_arr))
    return freeman_soln_image

mosaic_img = read_image('crayons.bmp')
original_image = read_image('crayons.jpg')
soln_image = get_freeman_solution_image(mosaic_img)
plt.imshow(soln_image)

pp_err, max_err = compute_errors(soln_image, original_image)
print("The average per-pixel error for crayons is: "+str(pp_err))
print("The maximum per-pixel error for crayons is: "+str(max_err))

mosaic_img = read_image('iceberg.bmp')
soln_image = get_freeman_solution_image(mosaic_img)
original_image = read_image('iceberg.jpg')
#plt.imshow(soln_image)
pp_err, max_err = compute_errors(soln_image, original_image)
print("The average per-pixel error for iceberg is: "+str(pp_err))
print("The maximum per-pixel error for iceberg is: "+str(max_err))
# Generate your solution image here and show it

mosaic_img = read_image('tony.bmp')
soln_image = get_freeman_solution_image(mosaic_img)
original_image = read_image('tony.jpg')
plt.imshow(soln_image)
pp_err, max_err = compute_errors(soln_image, original_image)
print("The average per-pixel error for iceberg is: "+str(pp_err))
print("The maximum per-pixel error for iceberg is: "+str(max_err))

"""### Mosaicing an Image
Now lets take a step backwards and mosaic an image.
"""

def get_mosaic_image(original_image):
    original_image = np.array(original_image)
    mosaic_image = np.zeros((np.shape(original_image)[0], np.shape(original_image)[1]))
    # Fetches green
    mosaic_image[0::2, 1::2] = original_image[0::2, 1::2, 1]
    mosaic_image[1::2, 0::2] = original_image[1::2, 0::2, 1]
    # Fetches red
    mosaic_image[0::2, 0::2] = original_image[0::2, 0::2, 0]
    # Fetches blue
    mosaic_image[1::2, 1::2] = original_image[1::2, 1::2, 2]
    return mosaic_image

colors_img = read_image('colors.jpg')
bw_colors = get_mosaic_image(colors_img)
cv2.imwrite("images/colors.bmp", bw_colors)
bw_colors = read_image('colors.bmp')
bw_colors = get_freeman_solution_image(bw_colors)
plt.imshow(bw_colors)
pp_err, max_err = compute_errors(bw_colors, colors_img)
print("The average per-pixel error for colors is: "+str(pp_err))
print("The maximum per-pixel error for colors is: "+str(max_err))

contrast_img = read_image('contrast.jpg')
bw_contrast = get_mosaic_image(contrast_img)
cv2.imwrite("images/contrast.bmp", bw_contrast)
bw_contrast = read_image('contrast.bmp')
bw_contrast = get_freeman_solution_image(bw_contrast)
pp_err, max_err = compute_errors(bw_contrast, contrast_img)
print("The average per-pixel error for contrast is: "+str(pp_err))
print("The maximum per-pixel error for contrast is: "+str(max_err))
plt.imshow(bw_contrast)

