import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageOps

#image = np.array(Image.open("C:/Users/aduttagu/Desktop/Bayesian-Matting-Implementation/input_training_lowres/GT06.png"))
#image_trimap = np.array(ImageOps.grayscale(Image.open("C:/Users/aduttagu/Desktop/Bayesian-Matting-Implementation/trimap_training_lowres/Trimap1/GT06.png")))


def Laplacianmatting(img, trimap):
    '''
    Description: This function calculates the alpha matte for an image given its trimap using Laplacian matting technique.

    Input:
    img: a numpy array representing the image.
    trimap: a numpy array representing the trimap of the image.

    Output:
    alpha: a numpy array representing the alpha matte of the image.

    '''

    # Convert img and trimap to double
    img = img.astype(np.float64) / 255.0
    trimap = trimap.astype(np.float64) / 255.0

    # Get the size of the image
    m, n, c = img.shape

    fg = trimap > 0.85
    bg = trimap < 0.15
    unk = ~(fg | bg)

    # Calculate the Laplacian matrix
    Laplacian = cv2.Laplacian(img, cv2.CV_64F)

    # Calculate the alpha matte
    alpha = np.zeros((m, n))

    for i in range(c):
        subLaplacian = Laplacian[:, :, i]
        alpha[unk[:, :]] += subLaplacian[unk[:, :]]**2

    alpha = 1 - np.sqrt(alpha/c)
    alpha[bg[:, :]] = 0
    alpha[fg[:, :]] = 1

    return alpha

#alpha = Laplacianmatting(image, image_trimap)
#plt.imshow(alpha, cmap='gray')
# plt.show()
