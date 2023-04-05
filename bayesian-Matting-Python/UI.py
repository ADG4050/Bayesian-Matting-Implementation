import PySimpleGUI as sg
import numpy as np
import cv2
import matplotlib.pyplot as plt
from Baysian_Mat import Bayesian_Matte
from PIL import Image, ImageOps
import time  # Execution TIme imports
import psutil


from laplac import Laplacianmatting
from compositing import compositing
from QualityTest import mse2d
from QualityTest import sad2d
from QualityTest import psnr2d
from smooth import smooth


# Import your Bayesian_Matte, Laplacianmatting, compositing, mse2d, sad2d, and psnr2d functions here


# Define the PySimpleGUI layout
layout = [
    [sg.Text("Select image file")],
    [sg.Input(key="-IMAGE_FILE-"), sg.FileBrowse()],
    [sg.Text("Select trimap file")],
    [sg.Input(key="-TRIMAP_FILE-"), sg.FileBrowse()],
    [sg.Text("Select GT file")],
    [sg.Input(key="-GT_FILE-"), sg.FileBrowse()],
    [sg.Button("Submit")],
    [sg.Output(size=(60, 2))]
]

# Create the PySimpleGUI window
window = sg.Window("Alpha Matte Calculation", layout)

# Start time for computing the execution time
st = time.time()

# Get initial memory usage
Memstart = psutil.Process().memory_info().rss / (1024 ** 2)

# Event loop
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    if event == "Submit":
        # Get the file paths from the input fields
        image_path = values["-IMAGE_FILE-"]
        trimap_path = values["-TRIMAP_FILE-"]
        gt_path = values["-GT_FILE-"]

        # Read the image, trimap, and GT files
        image = np.array(Image.open(image_path))
        image_trimap = np.array(Image.open(trimap_path))
        GT = np.array(Image.open(gt_path))

        # Step 2 : Calculating Bayesian Matte for the given trimap
        alpha, pixel_count = Bayesian_Matte(image, image_trimap)

        # Step 3 : Making it back to range (0-255) for display purpose
        alpha_disp = alpha * 255
        alpha_int8 = np.array(alpha, dtype=int)

        et = time.time()
        elapsed_time = et - st

        # Step 4 : End to End testing - 1 : Calculating the Laplacian Matting
        Lalpha = Laplacianmatting(image, image_trimap)

        # Step 5 : Compositing Function Display
        background = np.array(Image.open(
            'C:/Users/aduttagu/Desktop/Main/Bayesian-Matting-Implementation/bayesian-Matting-Python/background.png'))
        comp_Bay = compositing(image, alpha_disp, background)

        # Step 6 : Smoothening ALpha Methods
        smooth_alpha = smooth(alpha_disp)

        # Step 7 : Displaying THe Bayesian, Laplacian and GT.
        fig, axes = plt.subplots(nrows=2, ncols=2)
        axes[0, 0].imshow(alpha_disp, cmap='gray')
        axes[0, 0].set_title('Bayesian - Alpha Matte')
        axes[0, 1].imshow(Lalpha, cmap='gray')
        axes[0, 1].set_title('Laplacian - Alpha Matte')
        axes[1, 0].imshow(GT, cmap='gray')
        axes[1, 0].set_title('Ground Truth')
        axes[1, 1].imshow(smooth_alpha, cmap='gray')
        axes[1, 1].set_title('Smoothed Alpha')
        plt.show()


        plt.imshow(comp_Bay)
        plt.show()


# Close the PySimpleGUI window
window.close()




# Part of End to End testing - 1 : Performance Comparision between Laplacian and Bayesian.
Bay_MSE = mse2d(alpha_disp, GT)
Lap_MSE = mse2d(Lalpha, GT)
print("The MSE between the Ground Truth and Bayesian Alpha Matte is :", Bay_MSE)
print("The MSE between the Ground Truth and Laplacian Alpha Matte is :", Lap_MSE)

Bay_SAD = sad2d(alpha_disp, GT)
Lap_SAD = sad2d(Lalpha, GT)
print("The SAD between the Ground Truth and Bayesian Alpha Matte is :", Bay_SAD)
print("The SAD between the Ground Truth and Laplacian Alpha Matte is :", Lap_SAD)

Bay_PSNR = psnr2d(alpha_disp, GT)
Lap_PSNR = psnr2d(Lalpha, GT)
print("The PSNR between the Ground Truth and Bayesian Alpha Matte is :", Bay_PSNR)
print("The PSNR between the Ground Truth and Laplacian Alpha Matte is :", Lap_PSNR)


print('Execution time for Bayesian Matting: {:.3f} seconds'.format(
elapsed_time))

# get usage after completion of code
Memend = psutil.Process().memory_info().rss / (1024 ** 2)
Memuse = Memend - Memstart
print("Total memory consumed in execution of this program : ", Memuse, "MB's")
