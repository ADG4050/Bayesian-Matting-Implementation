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
    # calculate half of the shape for indexing
    m, n = [(ss - 1) / 2. for ss in shape]
    # create a meshgrid of x and y values using the shape
    y, x = np.ogrid[-m: m + 1, -n: n + 1]
    # calculate the exponential factor of the Gaussian function
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    # set values close to 0 to exactly 0 to improve performance
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    # normalize the kernel so it sums to 1
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    # return the kernel
    return h


def calcsurr_alpha(m, x, y, N):
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

    # Get the height, width, and number of channels of the image
    h, w, c = m.shape
    # Calculate the half of the window size
    halfN = N // 2
    # Initialize an empty window of size (N, N, c)
    r = np.zeros((N, N, c))
    # Calculate the indices of the sub-image to be extracted
    xmin = max(0, x - halfN)
    xmax = min(w, x + halfN + 1)
    ymin = max(0, y - halfN)
    ymax = min(h, y + halfN + 1)
    # Extract the sub-image and store it in the window
    r[halfN - (y - ymin): halfN + (ymax - y), halfN - (x - xmin)
               : halfN + (xmax - x)] = m[ymin: ymax, xmin: xmax]
    # Return the window
    return r

# @jit(nopython=True, cache=True)


def eqntn(mu_F, Sigma_F, mu_B, Sigma_B, C, Sigma_C, alpha_init, maxIter=50, minLike=1e-6):
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
# Initializing Matrices
    I = np.eye(3)
    F_Max = np.zeros(3)
    B_Max = np.zeros(3)
    alpha_Max = np.zeros(1)
    maxlike = -np.inf

    invsgma2 = 1/Sigma_C ** 2

    # These lines set up a nested loop structure. The outer loop iterates over the rows of the mu_F array. For each row, it sets mu_Fi to the current row,
    #  and invSigma_Fi to the inverse of the corresponding row of Sigma_F. The inner loop iterates over the rows of the mu_B array, and sets mu_Bj and
    # invSigma_Bj in a similar way. It also initializes several variables for the inner loop: alpha is set to alpha_init, myiter is set to 1, and lastLike is set
    # to a very large negative number.

    for i in range(mu_F.shape[0]):
        # Mean of Foreground pixel can have multiple possible values, iterating for all.
        mu_Fi = mu_F[i]
        invSigma_Fi = np.linalg.inv(Sigma_F[i])

        for j in range(mu_B.shape[0]):
            # Similarly, multiple mean values be possible for background pixel.
            mu_Bj = mu_B[j]
            invSigma_Bj = np.linalg.inv(Sigma_B[j])

            alpha = alpha_init
            myiter = 1
            lastLike = -1.7977e+308

            # Solving Minimum likelihood through numerical methods
            while True:
                # Making Equations for AX = b, where we solve for X.abs
                # X here has 3 values of forground pixel (RGB) and 3 values for background
                A = np.zeros((6, 6))
                A[:3, :3] = invSigma_Fi + I * alpha ** 2 * invsgma2
                A[:3, 3:] = A[3:, :3] = I * alpha * (1 - alpha) * invsgma2
                A[3:, 3:] = invSigma_Bj + I * (1 - alpha) ** 2 * invsgma2

                b = np.zeros((6, 1))
                b[:3] = np.reshape(invSigma_Fi @ mu_Fi + C *
                                   (alpha) * invsgma2, (3, 1))
                b[3:] = np.reshape(invSigma_Bj @ mu_Bj + C *
                                   (1-alpha) * invsgma2, (3, 1))

                # Solving for X and storing values for Forground and Background Pixels
                X = np.linalg.solve(A, b)
                F = np.maximum(0, np.minimum(1, X[0:3]))
                B = np.maximum(0, np.minimum(1, X[3:6]))

                # Solving for value of alpha once F and B are calculated
                alpha = np.maximum(0, np.minimum(
                    1, ((np.atleast_2d(C).T - B).T @ (F - B)) / np.sum((F - B)**2)))[0, 0]

                # Calculating likelihood value for
                like_C = - np.sum((np.atleast_2d(C).T -
                                  alpha*F - (1 - alpha) * B) ** 2) * invsgma2
                like_fg = (- ((F - np.atleast_2d(mu_Fi).T).T @
                           invSigma_Fi @ (F - np.atleast_2d(mu_Fi).T)) / 2)[0, 0]
                like_bg = (- ((B - np.atleast_2d(mu_Bj).T).T @
                           invSigma_Bj @ (B - np.atleast_2d(mu_Bj).T)) / 2)[0, 0]
                like = (like_C + like_fg + like_bg)

                if like > maxlike:
                    alpha_Max = alpha
                    maxLike = like
                    F_Max = F.ravel()
                    B_Max = B.ravel()

                if myiter >= maxIter or abs(like-lastLike) <= minLike:
                    break

                lastLike = like
                myiter += 1
    return F_Max, B_Max, alpha_Max


def Bayesian_Matte(img, trimap, N, MaxN=405, sigma=8, minN=10):
    '''
    Input:
    img: a numpy array representing the input image with dimensions (height, width, channels)
    trimap: a numpy array representing the trimap of the input image with dimensions (height, width)
    sigma: an integer value (default 8) representing the standard deviation of the Gaussian function used for calculating the weights of the unknown pixels
    N: an integer value (default 125) representing the size of the window used for calculating the weights of the surrounding pixels
    minN: an integer value (default 10) representing the minimum number of surrounding pixels needed for clustering
    N_max: an integer value (default 405) representing the maximum window size used when increasing the window size due to insufficient surrounding pixels for clustering

    Output:
    alpha: a numpy array representing the computed matte image with dimensions (height, width)

    Basic Function:
    The function computes the matte image using the Bayesian matting algorithm. It initializes an alpha matrix of zeros with the same dimensions as the input image, 
    creates separate logical matrices for the foreground, background, and unknown regions of the trimap, creates three channels for the foreground and background 
    matrices, and sets the alpha values for the foreground and unknown regions. It then iterates through the unknown region using a while loop, erodes the border 
    of the unknown region, calculates the weights of the surrounding pixels using a Gaussian function, clusters the foreground and background pixels, calculates 
    the alpha values for the unknown pixels, and updates the alpha matrix. The function returns the computed matte image.
    '''

    # Step 1 : Images are converted to float, so that all operations can be performed and then converted to (0-1) from (0-255), with h, w, c being image dimensions
    img = np.array(img, dtype='float')
    trimap = np.array(trimap, dtype='float')
    img = img / 255
    trimap = trimap / 255
    h, w, c = img.shape

    # Step 2: As per the bayesian matting paper, guassian falloff is used for weighting each pixel neighborhood range set earlier.
    # This creates a gaussian filter of size N x N, that can be applied to the image, given the image follows normal distribution.
    gaussian_wght = matlab_style_gauss2d((N, N), sigma)
    gaussian_wght /= np.max(gaussian_wght)

    # Step 3 : Here, FG_A and BG_A represent the foreground and background regions of an image based on a provided trimap. First, the foreground region is
    # defined by selecting pixels in the trimap with a value of 1 and assigning the corresponding pixels in the original image to FG_A. Similarly,
    # the background region is defined by selecting pixels with a value of 0 in the trimap and assigning the corresponding pixels in the original image to BG_A.
    fg_reg = trimap == 1
    FG_A = np.zeros((h, w, c))
    FG_A = img * np.reshape(fg_reg, (h, w, 1))
    bg_reg = trimap == 0
    BG_A = np.zeros((h, w, c))
    BG_A = img * np.reshape(bg_reg, (h, w, 1))

    # Step 4 : Creating only FG areas = 1 & unknown region = NaN in trimap, and creating new matrix for Foreground, Backgorund and Alpha for seperate computations .
    unkmask = np.logical_or(fg_reg, bg_reg) == False
    Al_pha = np.zeros(unkmask.shape)
    Al_pha[fg_reg] = 1
    Al_pha[unkmask] = np.nan

    # Step 5 : - Creating a sum of all the unknown pixels
    n_unknown = np.sum(unkmask)

    # Step 6 : Finding and storing the pixel values that have not been solved in a 3D Array, with first and second column having x and y co-ordinate and third column having
    # logical visiting status.
    Xx, Yy = np.where(unkmask == True)
    remain_n = np.vstack((Xx, Yy, np.zeros(Xx.shape))).T

    # Step 7 : While loop running for all unknown pixels that are solvable.
    while (sum(remain_n[:, 2]) != n_unknown):
        last_n = sum(remain_n[:, 2])

        # iterating for all pixels in the solvable range
        for i in range(n_unknown):
            # checking if solved or not
            if remain_n[i, 2] == 1:
                continue

            # If not solved, we try to solve
            else:
                # We get the location of the unsolved pixel
                y, x = map(int, remain_n[i, :2])

                # taking the surrounding pixel values around that pixel. (Used get_window Function to calaculate surrounding values)
                a_window = calcsurr_alpha(
                    Al_pha[:, :, np.newaxis], x, y, N)[:, :, 0]

                # Taking the surrounding pixel values around that pixel in foreground region. (Used get_window Function to calaculate surrounding values).
                # Then Calculating weights of that pixel = unknown pixel matrix squared X gausian matrix and then is structured as required.
                fg_window = calcsurr_alpha(FG_A, x, y, N)
                fg_weights = np.reshape(a_window**2 * gaussian_wght, -1)
                values_to_keep = np.nan_to_num(fg_weights) > 0
                fg_pixels = np.reshape(fg_window, (-1, 3))[values_to_keep, :]
                fg_weights = fg_weights[values_to_keep]

                # taking the surrounding pixel values around that pixel in background region. (Used calcsurr_alpha Function to calaculate surrounding values.
                # Then Calculating weights of that pixel = (1 - unknown pixel matrix) squared X gausian matrix and then is structured as required.
                bg_window = calcsurr_alpha(BG_A, x, y, N)
                bg_weights = np.reshape((1-a_window)**2 * gaussian_wght, -1)
                values_to_keep = np.nan_to_num(bg_weights) > 0
                bg_pixels = np.reshape(bg_window, (-1, 3))[values_to_keep, :]
                bg_weights = bg_weights[values_to_keep]

                # Clustering function will be done only weights of F & B are more than 10 or else we go to next iteration without clustering.
                if len(bg_weights) < minN or len(fg_weights) < minN:
                    continue

                # If enough pixels, Partitioning the foreground and background pixels (in a weighted manner with a clustering function) and calculate
                # the mean and variance of the cluster.
                mean_fg, cov_fg = clustFunc(fg_pixels, fg_weights)
                mean_bg, cov_bg = clustFunc(bg_pixels, bg_weights)
                alpha_init = np.nanmean(a_window.ravel())

                # We try to solve our 3 equation 7 variable problem with minimum likelihood estimation
                fg_pred, bg_pred, alpha_pred = eqntn(
                    mean_fg, cov_fg, mean_bg, cov_bg, img[y, x], 0.7, alpha_init)

                # Assigning the F, B and alpha values, Removing from unkowns for the next iteration.
                FG_A[y, x] = fg_pred.ravel()
                BG_A[y, x] = bg_pred.ravel()
                Al_pha[y, x] = alpha_pred
                remain_n[i, 2] = 1

                if (np.sum(remain_n[:, 2]) % 1000 == 0):
                    print("Successfully Calculated : {} out of unknown Pixels : {}.".format(
                        np.sum(remain_n[:, 2]), len(remain_n)))

        # The remaining pixel values which are unsolved are passed again with increasing window size with a Max window limit given.
        if sum(remain_n[:, 2]) == last_n:
            N = N + 10
            if (N == MaxN):
                n_unknown = sum(remain_n[:, 2])

            gaussian_wght = matlab_style_gauss2d((N, N), sigma)
            gaussian_wght /= np.max(gaussian_wght)
            print(N)

    return Al_pha, n_unknown
