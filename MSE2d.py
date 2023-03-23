import numpy as np
import math

def mse2d(alpha, GTalpha):
    s= len(alpha)
    i= 0
    diff= [0]*s

    while i<s:
        diff[i] = (alpha[i] - GTalpha[i])**2
        i=i+1
    msevalue = np.mean(diff)

    return msevalue


def sad2d(alpha, GTalpha):
    l= len(alpha)
    i= 0
    diff= [0]*l
    while i<s:
        diff[i] = abs(alpha[i] - GTalpha[i])
        i=i+1
    sadvalue = sum(diff)

    return sadvalue


def psnr2d(alpha, GTalpha):
    msevalue = mse2d(alpha, GTalpha)
    max_pixel = max(GTalpha)
    pnsrvalue = 10 * math.log10((max_pixel**2) / msevalue)
    return pnsrvalue



