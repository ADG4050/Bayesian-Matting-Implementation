#Please install the following library:
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt


# 1.This is a function used to calculate Mean squared error (MAE) of alpha, 
# where the reference image must be a related gound truth alpha. 
def mse2d(alpha, GTalpha):
    '''
    Takes in two 2d array, returns the MSE of them. 
    
    Args:
        alpha(float): the value of your alpha
        GTalpha(float): the value of ground truth
    
    Returns:
        msevalue(float): Returns the value of MSE
    '''
    # Get the number of rows
    s1= np.size(alpha,0)
    # Get the number of columns
    s2= np.size(alpha,1)
      
    # Set initial parameters.
    i=0
    j=0
    sum=0

    # Main loop: Subtract the corresponding positions of the two arrays. 
    # Then square the difference. 
    # The squared values of each pixel are summed and averaged.
    for i in range(s1):
        for j in range(s2):
            diff = (alpha[i][j] - GTalpha[i][j])**2
            sum=sum+diff
    msevalue=sum/(s1*s2)
    
    return msevalue

# 2.This is a function used to calculate  the sum of absolute differences (ASD) 
# between your alpha and a ground truth alpha, 
# where the reference image must be a related gound truth alpha. 
def sad2d(alpha, GTalpha):
    '''
    Takes in two 2d array, returns the SAD of them. 
    
    Args:
        alpha(float): the value of your alpha
        GTalpha(float): the value of ground truth
    
    Returns:
        sadvalue(float): Returns the value of SAD
    '''
    # Get the number of rows
    s1= np.size(alpha,0)
    # Get the number of columns
    s2= np.size(alpha,1)
      
    # Set initial parameters.
    i=0
    j=0
    sum=0

    # Main loop: Subtract the corresponding positions of the two arrays.
    # Then calculate the absolute value.
    # Sum all absolute values.
    for i in range(s1):
        for j in range(s2):
            diff =abs(alpha[i][j] - GTalpha[i][j])
            sum = sum + diff
    sadvalue = sum

    return sadvalue

# 3.This is a function used to calculate  the peak signal-to-noise ratio (PSNR) 
# between your alpha and a ground truth alpha, 
# where the reference image must be a related gound truth alpha. 

def psnr2d(alpha, GTalpha):
    '''
    Takes in two 2d array, returns the SAD of them. 
    
    Args:
        alpha(float): the value of your alpha
        GTalpha(float): the value of ground truth
    
    Returns:
        psnrvalue(float): Returns the value of PSNR,unit is decibel.
    '''
    # get MSE 
    msevalue = mse2d(alpha, GTalpha)
    # pixel maximum
    max_pixel = 255
    # calculate psnr
    pnsrvalue = 10 * math.log10((max_pixel**2) / msevalue)
    
    return pnsrvalue

# 4.This is a function used to show histograms of input image and composited image 
def histshow(inimage, compositedImage):
    '''
    Takes in two 2d array, plot the histogram of them. 
    
    Args:
        inimage(float): input image (BRG)
        compositedImage(float): composited image (BGR)
    
    Returns:
        plot two histograms in one figure.
    '''
    # convert BGR to YUV
    inimage1= cv2.cvtColor(inimage, cv2.COLOR_BGR2YUV) 
    compositedImage1=cv2.cvtColor(compositedImage, cv2.COLOR_BGR2YUV)
    
    # calculate histogram only using Y channel 
    hist_in = cv2.calcHist([inimage1],[0],None,[256],[0,256])
    hist_co = cv2.calcHist([compositedImage1],[0],None,[256],[0,256]) 
    
    # plot the above computed histogram
    plt.figure(1)    
    plt.subplot(1, 2, 1)
    plt.plot(hist_in, color='b')
    plt.title('Histogram of input image ')

    plt.figure(1)    
    plt.subplot(1, 2, 2)
    plt.plot(hist_co, color='b')
    plt.title('Histogram of composited Image')
    plt.show()

    
