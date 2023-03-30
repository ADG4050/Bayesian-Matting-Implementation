import numpy as np
import math

def mse2d(alpha, GTalpha):
    s1= np.size(alpha,0)
    s2= np.size(alpha,1)
    i=0
    j=0
    sum=0

    for i in range(s1):
        for j in range(s2):
            diff = (alpha[i][j] - GTalpha[i][j])**2
            sum=sum+diff

    msevalue=sum/(s1*s2)

    return msevalue

def sad2d(alpha, GTalpha):
    s1= np.size(alpha,0)
    s2= np.size(alpha,1)
    i=0
    j=0
    sum=0

    for i in range(s1):
        for j in range(s2):
            diff =abs(alpha[i][j] - GTalpha[i][j])
            sum = sum + diff
    sadvalue = sum

    return sadvalue

def psnr2d(alpha, GTalpha):
    msevalue = mse2d(alpha, GTalpha)
    max_pixel = 256
    pnsrvalue = 10 * math.log10((max_pixel**2) / msevalue)
    return pnsrvalue



