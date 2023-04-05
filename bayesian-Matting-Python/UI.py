import argparse
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

# create parser object
parser = argparse.ArgumentParser(description='Bayesian Matting')

# add arguments
parser.add_argument('--image', type=str, help='path to input image file')
parser.add_argument('--trimap', type=str, help='path to trimap image file')
parser.add_argument('--gt', type=str, help='path to ground truth image file')
parser.add_argument('--method', type=str, default='bayesian', choices=['bayesian', 'laplacian'], help='method for calculating alpha matte (default: bayesian)')
parser.add_argument('--smooth', action='store_true', help='flag for smoothing alpha matte')

# parse arguments
args = parser.parse_args()

# read input image, trimap, and ground truth
image = np.array(Image.open("C:/Users/aduttagu/Desktop/Main/Bayesian-Matting-Implementation/Image Dataset/input_training_lowres/GT01.png"))
image_trimap = np.array(ImageOps.grayscale(Image.open("C:/Users/aduttagu/Desktop/Main/Bayesian-Matting-Implementation/Image Dataset/trimap_training_lowres/Trimap1/GT01.png")))
GT = np.array(ImageOps.grayscale(Image.open("C:/Users/aduttagu/Desktop/Main/Bayesian-Matting-Implementation/Image Dataset/gt_training_lowres/GT01.png")))

# choose method for calculating alpha matte
if args.method == 'bayesian':
    alpha, pixel_count = Bayesian_Matte(image, image_trimap)
else:
    alpha = Laplacianmatting(image, image_trimap)

# smooth alpha matte if flag is True
if args.smooth:
    alpha = smooth(alpha * 255)

# display alpha matte, ground truth, and composite image
alpha_disp = alpha * 255
alpha_int8 = np.array(alpha, dtype=int)
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0,0].imshow(alpha_disp, cmap='gray')
axes[0,0].set_title('Alpha Matte')
axes[0,1].imshow(GT, cmap='gray')
axes[0,1].set_title('Ground Truth')
background = np.array(Image.open('background.png'))
comp = compositing(image, alpha_disp, background)
axes[1,0].imshow(comp)
axes[1,0].set_title('Composite Image')
plt.show()
