import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageOps


#image = np.array(Image.open("C:/Users/aduttagu/Desktop/Bayesian-Matting-Implementation/input_training_lowres/GT06.png"))
#image_trimap = np.array(ImageOps.grayscale(Image.open("C:/Users/aduttagu/Desktop/Bayesian-Matting-Implementation/trimap_training_lowres/Trimap1/GT06.png")))
#background = np.array(Image.open('C:/Users/aduttagu/Desktop/Bayesian-Matting-Implementation/background.png'))


def compositing(img, alpha, background): 

    '''
    Args:
    img (numpy.ndarray): A 3D numpy array representing the foreground image with dimensions (height, width, channels).
    alpha (numpy.ndarray): A 2D numpy array representing the alpha matte with dimensions (height, width).
    background (numpy.ndarray): A 3D numpy array representing the background image with dimensions (height, width, channels).

    Returns:
    numpy.ndarray: A 3D numpy array representing the composited image with dimensions (height, width, channels).
    '''

    H = alpha.shape[0]
    W = alpha.shape[1]

    # Resizing the background image to the size of the alpha channel
    background = cv2.resize(background, (W, H))

    # Converting the images to float
    img = img / 255
    alpha = alpha / 255
    background = background / 255

    # Reshaping the alpha channel to the size of the foreground image
    alpha = alpha.reshape((H, W, 1))
    alpha = np.broadcast_to(alpha, (H, W, 3))

    # Compositing the foreground and background images
    comp = img * (alpha) + background * (1 - alpha)

    return comp


#comp = compositing(image, image_trimap,background)
#plt.imshow(comp)
#plt.show()




