import numpy as np
import cv2
import matplotlib.pyplot as plt
from Baysian_Mat import bayesian_matte

from laplac import Laplacianmatting
from compositing import compositing
from QualityTest import mse2d
from QualityTest import sad2d
from QualityTest import psnr2d

#Step - Reading the input image, trimap and GT - alpha matte of that image.
img = cv2.imread("C:/Users/aduttagu/Desktop/Bayesian-Matting-Implementation/input_training_lowres/GT06.png")[:, :, :3]
trimap = cv2.imread("C:/Users/aduttagu/Desktop/Bayesian-Matting-Implementation/trimap_training_lowres/Trimap1/GT06.png")
GT = cv2.imread("C:/Users/aduttagu/Desktop/Bayesian-Matting-Implementation/gt_training_lowres/GT06.png")
background = cv2.imread('C:/Users/aduttagu/Desktop/Bayesian-Matting-Implementation/background.png')

Lalpha = Laplacianmatting(img, trimap)

trimap = trimap[:, :, 0]
GT = GT[:, :, 0]

#Step - Displaying the Alpha Matte and Ground Truth after calculating ALpha, Refer Function above for details
alpha = bayesian_matte(img, trimap)
    

# fig, axes = plt.subplots(nrows=1, ncols=3)
# axes[0].imshow(alpha, cmap='gray')
# axes[0].set_title('Bayesian - Alpha Matte')
# axes[1].imshow(Lalpha, cmap='gray')
# axes[1].set_title('Laplacian - Alpha Matte')
# axes[2].imshow(GT, cmap='gray')
# axes[2].set_title('Ground Truth')
# plt.show()


img = img / 255
alpha = trimap / 255
background = background / 255


resized_back = cv2.resize(background, None, fx=(img.shape[1]/background.shape[1]), fy=(img.shape[0]/background.shape[0]), interpolation=cv2.INTER_AREA)

Sh1 = alpha.shape[0]
Sh2 = alpha.shape[1]

alpha = alpha.reshape((Sh1, Sh2, 1))
alpha = np.broadcast_to(alpha, (Sh1, Sh2, 3))

comp = img * (alpha) + resized_back * (1 - alpha)

plt.imshow(comp)
plt.show()




# Comp_Bay = compositing(img, alpha, background)
# Comp_Lap = compositing(img, Lalpha, background)

# fig, axes = plt.subplots(nrows=1, ncols=2)
# axes[0].imshow(Comp_Bay, cmap='gray')
# axes[0].set_title('Bayesian - Composite')
# axes[1].imshow(Comp_Lap, cmap='gray')
# axes[1].set_title('Laplacian - Composite')
# plt.show()




# Bay_MSE = mse2d(alpha, GT)
# Lap_MSE = mse2d(Lalpha, GT)
# print("The MSE between the Ground Truth and Bayesian Alpha Matte is :", Bay_MSE)
# print("The MSE between the Ground Truth and Laplacian Alpha Matte is :", Lap_MSE)

# Bay_SAD = sad2d(alpha, GT)
# Lap_SAD = sad2d(Lalpha, GT)
# print("The SAD between the Ground Truth and Bayesian Alpha Matte is :", Bay_SAD)
# print("The SAD between the Ground Truth and Laplacian Alpha Matte is :", Lap_SAD)

# Bay_PSNR = psnr2d(alpha, GT)
# Lap_PSNR = psnr2d(Lalpha, GT)
# print("The PSNR between the Ground Truth and Bayesian Alpha Matte is :", Bay_PSNR)
# print("The PSNR between the Ground Truth and Laplacian Alpha Matte is :", Lap_PSNR)




    

