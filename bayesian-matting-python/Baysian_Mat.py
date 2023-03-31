# 5c22 Computational Method Assignment - Implementation of Bayesian Matting
# Author - Group : Yin Yang Artistic Chaos (Yuning, Abhishek, ChaiJie)

import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
from numba import jit 
from PIL import Image 
import PIL 
import matplotlib.pyplot as plt

from orchard_bouman_clust import clustFunc

def matlab_style_gauss2d(shape=(3, 3), sigma=0.5):
    """
    Input:
    shape: tuple of integers representing the dimensions of the Gaussian filter mask to be created (default: (3,3))
    sigma: standard deviation of the Gaussian distribution (default: 0.5)
    
    Output:
    Returns a 2D Gaussian filter mask with the specified shape and sigma value

    Basic Function:
    This function generates a 2D Gaussian filter mask with the specified shape and sigma value, which can be used for image processing tasks such as 
    blurring, smoothing, and noise reduction. The function implements the same algorithm as MATLAB's fspecial('gaussian',[shape],[sigma]) function, and 
    therefore produces similar results. The function computes the mask by first creating a coordinate grid using the specified shape, and then computing 
    the Gaussian function using the coordinates and the specified sigma value. The resulting filter mask is normalized to ensure that the sum of its elements
    equals 1.

    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y)/(2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


@jit(nopython=True, cache=True)
def get_window(m, x, y, N):
    '''
    Input:
    m: a numpy array of shape (h, w, c) representing an image.
    x: an integer representing the x-coordinate of the center pixel of the patch to be extracted from the image.
    y: an integer representing the y-coordinate of the center pixel of the patch to be extracted from the image.
    N: an integer representing the size of the patch to be extracted.
    
    Output:
    r: a numpy array of shape (N, N, c) representing the extracted patch from the image.
    
    Basic Function:
    This function extracts a patch of size N x N from an image m centered at the pixel (x, y). It first calculates the minimum and maximum x and y 
    coordinates to ensure that the patch does not extend outside of the image boundaries. It then calculates the coordinates for the patch within the output 
    array r. Finally, it extracts the patch from the input image m and returns it as the output array r.

    '''
    h, w, c = m.shape
    halfN = N//2
    r = np.zeros((N, N, c))
    xmin = max(0, x - halfN); xmax = min(w, x + (halfN+1))
    ymin = max(0, y - halfN); ymax = min(h, y + (halfN+1))
    pxmin = halfN - (x-xmin); pxmax = halfN + (xmax-x)
    pymin = halfN - (y-ymin); pymax = halfN + (ymax-y)

    r[pymin:pymax, pxmin:pxmax] = m[ymin:ymax, xmin:xmax]
    return r


@jit(nopython=True, cache=True)
def solve(mu_F, Sigma_F, mu_B, Sigma_B, C, sigma_C, alpha_init, maxIter, minLike):
    '''
    Estimates the foreground (F), background (B), and alpha values for a given input image Chan using a Gaussian mixture model.
    
    Inputs:
    mu_F (np.ndarray): Mean vector for the foreground Gaussian mixture model. Shape (n,3) where n is the number of foreground components.
    Sigma_F (np.ndarray): Covariance matrix for the foreground Gaussian mixture model. Shape (n,3,3) where n is the number of foreground components.
    mu_B (np.ndarray): Mean vector for the background Gaussian mixture model. Shape (m,3) where m is the number of background components.
    Sigma_B (np.ndarray): Covariance matrix for the background Gaussian mixture model. Shape (m,3,3) where m is the number of background components.
    Chan (np.ndarray): Input image. Shape (h,w,3) where h and w are the height and width of the image, respectively.
    sigma_C (float): Standard deviation for the observation model.
    alpha_init (float): Initial value for the alpha channel. Value should be between 0 and 1.
    maxIter (int): Maximum number of iterations allowed for the algorithm.
    minLike (float): Minimum change in likelihood between consecutive iterations allowed for the algorithm to terminate.
    
    Outputs:
    F_Max (np.ndarray): Estimated foreground values. Shape (3,).
    B_Max (np.ndarray): Estimated background values. Shape (3,).
    alpha_Max (float): Estimated alpha channel value. Value should be between 0 and 1.
    
    Basic Function:
    The function estimates the foreground (F), background (B), and alpha values for a given input image Chan using a Gaussian mixture model. 
    The function loops through each foreground and background cluster to estimate their respective means and inverse covariance matrices. 
    The alpha channel value is initialized to alpha_init and a loop is started that continues until the maximum number of iterations has 
    been reached or the change in likelihood between consecutive iterations is below a certain threshold. The foreground and background values 
    are estimated using the current alpha value and the given input values. The alpha channel value is estimated using the current F and B 
    values by solving for alpha using a formula based on the pixel value Chan, the foreground value F, and the background value B. The 
    likelihood is calculated for the observation (C) as well as for the foreground (F) and background (B) estimates. The estimated F, B, 
    and alpha values with the maximum likelihood are returned.

    '''
     # Step : Creating a 3x3 identity matrix I and an empty array vals to store estimated foreground, background, and alpha values and their likelihoods.
    I = np.eye(3)
    FMax = np.zeros(3)
    BMax = np.zeros(3)
    alphaMax = 0
    maxlike = - np.inf
    invsgma2 = 1/sigma_C**2

    # Step : Loops through each foreground and background cluster to estimate their respective means and inverse covariance matrices.
    for i in range(mu_F.shape[0]):
        mu_Fi = mu_F[i]
        invSigma_Fi = np.linalg.inv(Sigma_F[i])
        for j in range(mu_B.shape[0]):
            mu_Bj = mu_B[j]
            invSigma_Bj = np.linalg.inv(Sigma_B[j])

            # Initializing the alpha channel value to alpha_init, iteration counter to iter, and the last likelihood value to negative infinity
            alpha = alpha_init
            myiter = 1
            lastLike = -1.7977e+308


            # Begining a loop that will continue until the maximum number of iterations has been reached or the change in likelihood between 
            # consecutive iterations is below a certain threshold.
            while True:
                # solve for F,B
                # Estimating the foreground and background values using the current alpha value and the given input values. Solving for F and B by finding the 
                # least squares solution to the linear system A*X = b, where A is a 6x6 matrix, b is a 6x1 vector, and X is a 6x1 vector containing F and B. The 
                # values of F and B are then clamped to be between 0 and 1.
                A11 = invSigma_Fi + I*alpha**2 * invsgma2
                A12 = I*alpha*(1-alpha) * invsgma2
                A22 = invSigma_Bj+I*(1-alpha)**2 * invsgma2
                A = np.vstack((np.hstack((A11, A12)), np.hstack((A12, A22))))
                b1 = invSigma_Fi @ mu_Fi + C*(alpha) * invsgma2
                b2 = invSigma_Bj @ mu_Bj + C*(1-alpha) * invsgma2
                b = np.atleast_2d(np.concatenate((b1, b2))).T

                X = np.linalg.solve(A, b)
                F = np.maximum(0, np.minimum(1, X[0:3]))
                B = np.maximum(0, np.minimum(1, X[3:6]))
                
                # Estimating the alpha channel value using the current F and B values by solving for alpha using a formula based on the pixel value Chan, 
                # the foreground value F, and the background value B.

                alpha = np.maximum(0, np.minimum(1, ((np.atleast_2d(C).T-B).T @ (F-B))/np.sum((F-B)**2)))[0,0]
                
                # The likelihood is calculated for the observation (C) as well as for the foreground (F) and background (B) estimates. 
                # The likelihood for the observation is calculated using the sum of squared differences between the observed pixel and the weighted combination 
                # of foreground and background pixels, where alpha is the weight of the foreground pixels and (1-alpha) is the weight of the background pixels. 
                # The calculation is normalized by the covariance cov_C.
                L_C = - np.sum((np.atleast_2d(C).T -alpha*F-(1-alpha)*B)**2) * invsgma2
                L_F = (- ((F- np.atleast_2d(mu_Fi).T).T @ invSigma_Fi @ (F-np.atleast_2d(mu_Fi).T))/2)[0,0]
                L_B = (- ((B- np.atleast_2d(mu_Bj).T).T @ invSigma_Bj @ (B-np.atleast_2d(mu_Bj).T))/2)[0,0]
                like = (L_C + L_F + L_B)
               

                #The code then checks if the maximum number of iterations (maxIter) has been reached or if the change in likelihood from the previous 
                # iteration is smaller than a specified threshold (minLike). If either condition is met, the loop is terminated.
                if like > maxlike:
                    alphaMax = alpha
                    maxlike = like
                    FMax = F.ravel()
                    BMax = B.ravel()

                if myiter >= maxIter or abs(like-lastLike) <= minLike:
                    break

                lastLike = like
                myiter += 1

    return FMax, BMax, alphaMax


def bayesian_matte(img, trimap, sigma=8, N=125, minN=10, N_max = 175):
    '''
    Input:
    img: a numpy array representing the input image with dimensions (height, width, channels)
    trimap: a numpy array representing the trimap of the input image with dimensions (height, width)
    sigma: an integer value (default 8) representing the standard deviation of the Gaussian function used for calculating the weights of the unknown pixels
    N: an integer value (default 125) representing the size of the window used for calculating the weights of the surrounding pixels
    minN: an integer value (default 10) representing the minimum number of surrounding pixels needed for clustering
    N_max: an integer value (default 175) representing the maximum window size used when increasing the window size due to insufficient surrounding pixels for clustering

    Output:
    alpha: a numpy array representing the computed matte image with dimensions (height, width)
    
    Basic Function:
    The function computes the matte image using the Bayesian matting algorithm. It initializes an alpha matrix of zeros with the same dimensions as the input image, 
    creates separate logical matrices for the foreground, background, and unknown regions of the trimap, creates three channels for the foreground and background 
    matrices, and sets the alpha values for the foreground and unknown regions. It then iterates through the unknown region using a while loop, erodes the border 
    of the unknown region, calculates the weights of the surrounding pixels using a Gaussian function, clusters the foreground and background pixels, calculates 
    the alpha values for the unknown pixels, and updates the alpha matrix. The function returns the computed matte image.
    '''
    # Step 1 :- Taking the input image converting it into double and initializing alpha with image sizes
    im = img/255 
    h, w, c = im.shape 
    alpha = np.zeros((h, w))

    # Step 2 :- creates seperate logical matrix of foreground, backgound and unknown region
    fg_reg = trimap == 255
    bg_reg = trimap == 0
    unkmask = True ^ np.logical_or(fg_reg, bg_reg)

    # Step 3 :- Creating three channels for foreground and Background matrix 
    foreground = im*np.repeat(fg_reg[:, :, np.newaxis], 3, axis=2)
    background = im*np.repeat(bg_reg[:, :, np.newaxis], 3, axis=2)

    #Step 4 :- Creating only FG areas = 1 & unknown region = NaN in trimap, and creating new matrix for Foreground, Backgorund and Alpha for seperate computations 
    alpha[fg_reg] = 1
    F = np.zeros(im.shape)
    B = np.zeros(im.shape)
    Al_pha = np.zeros(trimap.shape)
    alpha[unkmask] = np.nan
    
    # Step 5 : - Creating a sum of all the unknown pixels
    Unknown_sum = np.sum(unkmask)
    unkreg = unkmask
    
    # While Loop parameters and window counters
    n = 1
    m_i = 0
    max_li = 10
    # Kernel is matrix which is used to erode the borders of the the unknown region
    kernel = np.ones((3, 3))

    while n < Unknown_sum:
        # Step 5.1 :- certain pixels are removed from the unknown region of the image to detect the edges of the image.
        unkreg = cv2.erode(unkreg.astype(np.uint8), kernel, iterations=1)
        unkpixels = np.logical_and(np.logical_not(unkreg), unkmask)

        # Step 5.2 :- As per the bayesian matting paper, guassian falloff is used for weighting each pixel neighborhood range set earlier.
        # This creates a gaussian filter of size N x N, that can be applied to the image, given the image follows normal distribution. 
        gaussian_weights = matlab_style_gauss2d((N, N), sigma)
        gaussian_weights = gaussian_weights/np.max(gaussian_weights)

        # Step 5.3 finding the position of the pixels inside the above unknown boundary region.
        Y, X = np.nonzero(unkpixels)
        
        max_rep = 0
        
        # Step 5.4 :- Creating a for loop inside the unknown region for iterating along the count of Y that we just computed.
        for i in range(Y.shape[0]):
            # creating a progress bar based on the number of unknown pixels
            if n % 100 == 0:
                print(n, Unknown_sum)

            # taking the pixel and reshaping its 3 channel value into a 3 X 1 matrix
            y, x = Y[i], X[i]
            p = im[y, x]
            
            # Step 5.5 :- taking the surrounding pixel values around that pixel. (Used calcsurr_alpha Function to calaculate surrounding values
            a = get_window(alpha[:, :, np.newaxis], x, y, N)[:, :, 0]


            # Step 5.6 :- Taking the surrounding pixel values around that pixel in foreground region. (Used calcsurr_alpha Function to calaculate surrounding values). 
            # Then Calculating weights of that pixel = unknown pixel matrix squared X gausian matrix and then is structured as required.
            f_pixels = get_window(foreground, x, y, N)
            f_weights = (a**2 * gaussian_weights).ravel()

            f_pixels = np.reshape(f_pixels, (N*N, 3))
            posInds = np.nan_to_num(f_weights) > 0
            f_pixels = f_pixels[posInds, :]
            f_weights = f_weights[posInds]

            # Step 5.7 :- taking the surrounding pixel values around that pixel in background region. (Used calcsurr_alpha Function to calaculate surrounding values. 
            # Then Calculating weights of that pixel = (1 - unknown pixel matrix) squared X gausian matrix and then is structured as required.
            b_pixels = get_window(background, x, y, N)
            b_weights = ((1-a)**2 * gaussian_weights).ravel()

            b_pixels = np.reshape(b_pixels, (N*N, 3))
            posInds = np.nan_to_num(b_weights) > 0
            b_pixels = b_pixels[posInds, :]
            b_weights = b_weights[posInds]

             # Step 5.8 :- Clustering function will be done only weights of F & B are more than 10 or else we go to next iteration without clustering.
            if len(f_weights) < minN or len(b_weights) < minN:
                max_rep = max_rep + 1; 
            # Step 5.8.1* If whole iteration in the for loop runs without getting enough background or foreground values for some time, 
            # means the loop has become infinite as no more suitable window to calculate F,B and alpha, hence condition to check the same thing again increased window size 
            # and break the loop and display the result accordingly.
                if (max_rep == len(Y)):
                    m_i = m_i + 1
                    if (m_i == max_li):
                        m_i = 0
                        N = N + 10
                        if (N == N_max):
                            n = Unknown_sum
                continue


            # Step 5.9 :- Partitioning the foreground and background pixels (in a weighted manner with a clustering function) and calculate the mean and variance of the cluster.
            mu_f, sigma_f = clustFunc(f_pixels, f_weights)
            mu_b, sigma_b = clustFunc(b_pixels, b_weights)

            # Step 5.10 :- Calculates the initial alpha value mean of the surrounding pixels.
            alpha_init = np.nanmean(a.ravel())

            # Step 5.11 :- Calculating the F, B and alpha values based on MAP and Bayes method.
            f, b, alphaT = solve(mu_f, sigma_f, mu_b, sigma_b, p, 0.01, alpha_init, 50, 1e-6)

            # Step 5.12 :- Assigning the F, B and alpha values.
            foreground[y, x] = f.ravel()
            background[y, x] = b.ravel()
            alpha[y, x] = alphaT
            
            # Step 5.13 :- Removing from unkowns for the next iteration.
            unkmask[y, x] = 0
            
            n += 1

    return alpha


