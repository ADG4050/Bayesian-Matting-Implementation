# 5c22 Computational Method Assignment - Implementation of Bayesian Matting
# Author - Group : Yin Yang Artistic Chaos (Yuning, Abhishek, ChaiJie)

import numpy as np
import cv2
import matplotlib.pyplot as plt
from Baysian_Mat import Bayesian_Matte
from PIL import Image, ImageOps
import time  # Execution TIme imports
import psutil


from laplac import Laplacianmatting
from compositing import compositing
from QualityTest import mse2d
from QualityTest import sad2d
from QualityTest import psnr2d
from smooth import smooth

# Step 1 : Read image, GT and trimap.
image = np.array(Image.open(
    "C:/Users/aduttagu/Desktop/Main/Bayesian-Matting-Implementation/Image Dataset/input_training_lowres/GT16.png"))
image_trimap = np.array(ImageOps.grayscale(Image.open(
    "C:/Users/aduttagu/Desktop/Main/Bayesian-Matting-Implementation/Image Dataset/trimap_training_lowres/Trimap1/GT16.png")))
GT = np.array(ImageOps.grayscale(Image.open(
    "C:/Users/aduttagu/Desktop/Main/Bayesian-Matting-Implementation/Image Dataset/gt_training_lowres/GT16.png")))


# Step 2 : Calculating Bayesian Matte for the given trimap
window_size = 75
N = window_size
alpha, pixel_count = Bayesian_Matte(image, image_trimap, N)

# Step 3 : Making it back to range (0-255) for display purpose
alpha_disp = alpha * 255
alpha_disp2 = alpha_disp
alpha_int8 = np.array(alpha, dtype=int)

plt.imshow(alpha_disp2, cmap = 'gray')
plt.show()

Bay_MSE = mse2d(alpha_disp2, GT)
print("The MSE between the Ground Truth and Bayesian Alpha Matte is :", Bay_MSE)


sm_alpha = alpha_disp
# Split the image into its RGB channels
r, g, b = cv2.split(image)
# Create a mask where the pixel values meet the specified criteria
mask = np.logical_and.reduce([(r >= 10) & (r <= 51), (g >= 20) & (g <= 52), (b >= 34) & (b <= 87)])
change_mat = np.zeros_like(image[:,:,0])
change_mat[139:476, 81:338] = mask[139:476, 81:338]
change_mat[138:155, 124:217] = 0
change_mat[293:308, 80:121] = 0

# Set all locations in the alpha matte to 1 where change_mat is 1
# Convert the change_mat to a logical data type
change_mat = change_mat.astype(bool)

# Set all locations in alpha to 1 where change_mat is true
sm_alpha[change_mat] = 255

plt.imshow(sm_alpha, cmap = 'gray')
plt.show()

smooth_MSE = mse2d(sm_alpha, GT)
print("The MSE between the Ground Truth and smoothed Matte is :", smooth_MSE)
