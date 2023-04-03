
import numpy as np
from PIL import Image
import math
import subprocess

def mse2d(alpha, GTalpha):
    s1= np.size(alpha,0)/255
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

def my_comparison():
    
    Lap_alpha = 'C:/Users/zhuy3/Documents/La/la_alpha.png'
    # Bay_alpha = 'C:/Users/zhuy3/Documents/La/bay_alpha.png'
    GT = 'C:/Users/zhuy3/Documents/La/groundtruth2.png'

    # image arrays
    Lap_alpha_array = np.array(Image.open(Lap_alpha))
    #Bay_alpha_array = np.array(Image.open(Bay_alpha))
    GT_array = np.array(Image.open(GT))

    # MSE between Lap_alpha and GT
    Lap_alpha_mse = mse2d(Lap_alpha_array[:, :, 0], GT_array[:, :, 0])

    #  MSE between Bay_alpha and GT
    #Bay_alpha_mse = mse2d(Bay_alpha_array[:, :, 0], GT_array[:, :, 0])

    print("MSE between Lap_alpha and GT:", Lap_alpha_mse)
    #print("MSE between Bay_alpha and GT:", Bay_alpha_mse)

if __name__ == "__main__":
    my_comparison()

subprocess.call(['python', 'e2e.py'])
