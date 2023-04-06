import numpy as np
import math


def mse2d(alpha, GT):
    """
    Calculate the Mean Squared Error (MSE) between two 2D images represented as numpy arrays.

    Parameters:
    alpha (numpy array): The first image to compare.
    GT (numpy array): The second image to compare.

    Returns:
    float: The MSE value between the two images.
    """
    if np.isnan(np.max(alpha)):
        msevalue = np.nansum(np.abs(alpha - GT)**2) / \
            (alpha.shape[0]*alpha.shape[1])
    else:
        msevalue = np.sum(np.abs(alpha - GT)**2) / \
            (alpha.shape[0]*alpha.shape[1])

    return msevalue


def sad2d(alpha, GT):
    """
    Calculate the Sum of Absolute Differences (SAD) between two 2D images represented as numpy arrays.

    Parameters:
    alpha (numpy array): The first image to compare.
    GT (numpy array): The second image to compare.

    Returns:
    float: The SAD value between the two images.
    """
    if np.isnan(np.max(alpha)):
        sadvalue = np.nansum(np.abs(alpha - GT))
    else:
        sadvalue = np.sum(np.abs(alpha - GT))

    return sadvalue


def psnr2d(alpha, GTalpha):
    """
    Calculate the Peak Signal to Noise Ratio (PSNR) between two 2D images represented as numpy arrays.

    Parameters:
    alpha (numpy array): The first image to compare.
    GTalpha (numpy array): The second image to compare.

    Returns:
    float: The PSNR value between the two images.
    """

    msevalue = mse2d(alpha, GTalpha)
    max_pixel = 255
    pnsrvalue = 10 * math.log10((max_pixel**2) / msevalue)
    return pnsrvalue
