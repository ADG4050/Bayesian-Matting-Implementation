import numpy as np
from skimage.color import rgb2gray
from skimage.filters import laplace

def Laplacianmatting(img, trimap):
    # Convert img and trimap to double
    img = img.astype(np.float64) / 255.0
    trimap = trimap.astype(np.float64) / 255.0
    
    # Get the size of the image
    m, n, c = img.shape
    
    # Calculate the foreground, background, and unknown pixels
    fg = trimap > 0.9
    bg = trimap < 0.1
    unk = ~(fg | bg)
    
    # Calculate the Laplacian matrix
    laplacian = laplace(rgb2gray(img))
    
    # Calculate the alpha matte
    alpha = np.zeros((m, n))
    
    for i in range(c):
        subLaplacian = laplacian[:,:,i]
        alpha[unk] += subLaplacian[unk] ** 2
    
    alpha = 1 - np.sqrt(alpha / c)
    alpha[bg[:,:,0]] = 0
    alpha[fg[:,:,0]] = 1
    
    return alpha
