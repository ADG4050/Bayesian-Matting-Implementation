import cv2
from matplotlib import pyplot as plt
import numpy as np


img_path = 'C:/Users/aduttagu/Desktop/Bayesian-Matting-Implementation/input_training_lowres/GT01.png'
alpha_path ='C:/Users/aduttagu/Desktop/Bayesian-Matting-Implementation/trimap_training_lowres/Trimap1/GT01.png'
background_path = 'C:/Users/aduttagu/Desktop/Bayesian-Matting-Implementation/background.png'



img = cv2.imread(img_path)
alpha = cv2.imread(alpha_path)[:,:,0]
background = cv2.imread(background_path)

def compositing(img, alpha, background): 

    img = img / 255
    alpha = alpha / 255
    background = background / 255


    resized_back = cv2.resize(background, None, fx=(img.shape[1]/background.shape[1]), fy=(img.shape[0]/background.shape[0]), interpolation=cv2.INTER_AREA)

    Sh1 = alpha.shape[0]
    Sh2 = alpha.shape[1]

    alpha = alpha.reshape((Sh1, Sh2, 1))
    alpha = np.broadcast_to(alpha, (Sh1, Sh2, 3))

    comp = img * (alpha) + resized_back * (1 - alpha)

    return comp


comp = compositing(img,alpha,background)
plt.imshow(comp)
plt.show()




