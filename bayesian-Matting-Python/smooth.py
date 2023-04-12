import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageOps


def smooth(alpha_matte):
    fg = alpha_matte > 125
    bg = alpha_matte < 75

    unkreg = fg
    kernel = np.ones((2, 2))
    unkreg2 = cv2.erode(unkreg.astype(np.uint8), kernel, iterations=1)
    border_pixels = np.logical_and(np.logical_not(unkreg2), unkreg)

    return unkreg2
