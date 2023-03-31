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

    imag1 = np.array(alpha) / 255
    imag2 = np.array(GT) / 255
    
    diff = imag1 - imag2
    msevalue = np.nanmean(diff**2)/(imag1.shape[0]*imag1.shape[1])
  
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

    imag1 = np.array(alpha) / 255
    imag2 = np.array(GT) / 255

    diff = np.abs(imag1 - imag2)
    sadvalue = np.sum(diff, axis=None, where=~np.isnan(diff))

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
    max_pixel = 256
    pnsrvalue = 10 * math.log10((max_pixel**2) / msevalue)
    return pnsrvalue


