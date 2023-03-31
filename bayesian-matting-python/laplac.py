import numpy as np
import cv2
from matplotlib import pyplot as plt

img_path = 'C:/Users/aduttagu/Desktop/Bayesian-Matting-Implementation/input_training_lowres/GT06.png'
trimap_path = 'C:/Users/aduttagu/Desktop/Bayesian-Matting-Implementation/trimap_training_lowres/Trimap1/GT06.png'

img = cv2.imread(img_path)
trimap = cv2.imread(trimap_path)

def Laplacianmatting(img, trimap):
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
        subLaplacian = Laplacian[:,:,i]
        alpha[unk[:,:,0]] += subLaplacian[unk[:,:,0]]**2
   
    alpha = 1 - np.sqrt(alpha/c)
    alpha[bg[:,:,0]] = 0
    alpha[fg[:,:,0]] = 1
   
    return alpha

alpha = Laplacianmatting(img, trimap)
plt.imshow(alpha, cmap='gray')
plt.show()



