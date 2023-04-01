# 5c22 Computational Method Assignment - Implementation of Bayesian Matting
# Author - Group : Yin Yang Artistic Chaos (Yuning, Abhishek, ChaiJie)

import numpy as np
import cv2
import matplotlib.pyplot as plt
from Baysian_Mat import Bayesian_Matte
from PIL import Image, ImageOps

from laplac import Laplacianmatting
from compositing import compositing
from QualityTest import mse2d
from QualityTest import sad2d
from QualityTest import psnr2d
from smooth import smooth

# Step 1 : Read image, GT and trimap.
image = np.array(Image.open("C:/Users/aduttagu/Desktop/Bayesian-Matting-Implementation/input_training_lowres/GT06.png"))
image_trimap = np.array(ImageOps.grayscale(Image.open("C:/Users/aduttagu/Desktop/Bayesian-Matting-Implementation/trimap_training_lowres/Trimap1/GT06.png")))
GT = np.array(ImageOps.grayscale(Image.open("C:/Users/aduttagu/Desktop/Bayesian-Matting-Implementation/gt_training_lowres/GT06.png")))

# Step 2 : Calculating Bayesian Matte for the given trimap
alpha,pixel_count = Bayesian_Matte(image,image_trimap) 

# Step 3 : Making it back to range (0-255) for display purpose
alpha_disp = alpha * 255
alpha_int8 = np.array(alpha,dtype = int)

# Step 4 : Unit Test - 1 : Calculating the Laplacian Matting
Lalpha = Laplacianmatting(image, image_trimap)

# Step 5 : Displaying THe Bayesian, Laplacian and GT.
fig, axes = plt.subplots(nrows = 1, ncols = 3)
axes[0].imshow(alpha_disp, cmap='gray')
axes[0].set_title('Bayesian - Alpha Matte')
axes[1].imshow(Lalpha, cmap='gray')
axes[1].set_title('Laplacian - Alpha Matte')
axes[2].imshow(GT, cmap='gray')
axes[2].set_title('Ground Truth')
plt.show()

# Step 6 : Compositing Function Display
background = np.array(Image.open('C:/Users/aduttagu/Desktop/Bayesian-Matting-Implementation/background.png'))
comp_Bay = compositing(image, alpha_disp, background)

plt.imshow(comp_Bay)
plt.show()

# Step 7 : Smoothening ALpha Methods
smooth_alpha = smooth(alpha_disp)

plt.imshow(smooth_alpha)
plt.show()
# fig, axes2 = plt.subplots(nrows = 1, ncols = 2)
# axes[0].imshow(alpha_disp, cmap = 'gray')
# axes[0].set_title('original Alpha Matte')
# axes[1].imshow(smooth_alpha, cmap = 'gray')
# axes[1].set_title('Eroded - Smoothed')


# Part of Unit test - 1 : Performance Comparision between Laplacian and Bayesian. 
Bay_MSE = mse2d(alpha_disp, GT)
Lap_MSE = mse2d(Lalpha, GT)
print("The MSE between the Ground Truth and Bayesian Alpha Matte is :", Bay_MSE)
print("The MSE between the Ground Truth and Laplacian Alpha Matte is :", Lap_MSE)

Bay_SAD = sad2d(alpha_disp, GT)
Lap_SAD = sad2d(Lalpha, GT)
print("The SAD between the Ground Truth and Bayesian Alpha Matte is :", Bay_SAD)
print("The SAD between the Ground Truth and Laplacian Alpha Matte is :", Lap_SAD)

Bay_PSNR = psnr2d(alpha_disp, GT)
Lap_PSNR = psnr2d(Lalpha, GT)
print("The PSNR between the Ground Truth and Bayesian Alpha Matte is :", Bay_PSNR)
print("The PSNR between the Ground Truth and Laplacian Alpha Matte is :", Lap_PSNR)

