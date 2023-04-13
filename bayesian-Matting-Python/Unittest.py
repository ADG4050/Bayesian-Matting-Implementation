
import unittest
from PIL import Image, ImageOps
from Baysian_Mat import Bayesian_Matte
import numpy as np


from compositing import compositing
from numpy.testing import assert_array_equal
from Baysian_Mat import calcsurr_alpha
from orchard_bouman_clust import clustFunc



image = np.array(Image.open(
    "C:/Users/aduttagu/Desktop/Main/Bayesian-Matting-Implementation/Image Dataset/input_training_lowres/GT02.png"))
image_trimap = np.array(ImageOps.grayscale(Image.open(
    "C:/Users/aduttagu/Desktop/Main/Bayesian-Matting-Implementation/Image Dataset/trimap_training_lowres/Trimap1/GT02.png")))
background = np.array(Image.open(
    'C:/Users/aduttagu/Desktop/Main/Bayesian-Matting-Implementation/bayesian-Matting-Python/background.png'))


alpha, pixel_count = Bayesian_Matte(image, image_trimap, N = 75)
alpha_disp = alpha * 255
alpha_int8 = np.array(alpha, dtype=int)

comp_Bay = compositing(image, alpha_disp, background)

# unittest Codes


class testcode(unittest.TestCase):
    """ Unit test - 1 : Checks Input Image is correct or not"""

    def test_dimcheck(self):
        # check if it is a 3 channel image
        s3 = np.size(image, 2)
        message = "Your input image must be 3 channels"
        self.assertEqual(s3, 3, message)

    """ Unit test - 2 : Checks Input and Composite image shape"""

    def test_image_dim(self):
        # Get the shapes of the input image and composite image
        Image1 = image.shape
        Image2 = comp_Bay.shape

        # Assert that the input image and composite image are equal
        self.assertEqual(
            Image1, Image2, "The shapes of the input image and composite are not equal")

    """ Unit test - 3 : Checks Input image and Alpha height and width"""

    def test_image_size(self):
        # Get the shapes of input image and alpha
        h1, w1, c1 = image.shape
        h2, w2 = alpha_disp.shape

        # Assert that the sizes are equal
        self.assertEqual(
            h1, h2, "Height of the input image and alpha matte are not equal")
        self.assertEqual(
            w1, w2, "Width of the input image and alpha matte are not equal")

    '''Unit test - 4 : Checks the window function is working or not.
    A random 100x100x3 image is created. The get_window() function is called with the center coordinates (25, 25) and the default window size of 75. 
    The expected output is a 75x75x3 sub-image of the input image centered at (25, 25). Again, the assert_array_equal() method is used to compare the 
    shape of the output of the function with the expected shape. '''

    def test_get_window(self):
        m = np.random.rand(100, 100, 3)
        window = calcsurr_alpha(m, 25, 25, 75)
        window_shape = window.shape
        expected_shape = (75, 75, 3)
        assert_array_equal(window_shape, expected_shape)

    '''Unit test - 5 : Checks the functionality of orchard Bouman function.
    Here we make random arrays S (100X3) and w (100X1) and pass it through the orchard Bouman function to confirm the shape and size of mean 
    and covariance matrices, which is checked through an assert syntax. Answer should be 3 and 3X3'''

    def test_clustFunc(self):
        # Generate random data and weights
        np.random.seed(1234)
        S = np.random.randn(100, 3)
        w = np.ones(100)
        mu, sigma = clustFunc(S, w)
        assert mu.shape[1] == 3
        assert sigma.shape[1:] == (3, 3)

        
    '''Unit test - 6 : Checks whether correct trimap provided is of the same input image only, even the dimensions match for the wrong trimap 
    and correct image.'''
    def test_trimap_is_corr(self):
        pic = image / 255

        def rgb2yuv(r, g, b):
        # Convert from rgb to yuv
            y = 0.3 * r + 0.6 * g + 0.1 * b
            u = -0.15 * r + 0.3 * g + 0.45 * b + 128
            v = 0.4375 * r - 0.3750 * g - 0.0625 * b + 128
            return y, u, v

        y, u, v = rgb2yuv(pic[:,:,0], pic[:,:,1], pic[:,:,2])

        # Set the centroid
        # Use the average and std deviation of a known patch
        sx = slice(0, 51)
        sy = slice(700, 750)

        u_bar = np.mean(u[sx, sy])
        u_var = 2 * (np.std(u[sx, sy], ddof=1))**2

        v_bar = np.mean(v[sx, sy])
        v_var = 2 * (np.std(v[sx, sy]))**2

        y_bar = np.mean(y[sx, sy])
        y_var = 2 * (np.std(y[sx, sy]))**2

        E_t = 40

        energy = ((y - y_bar)**2)/y_var + ((v - v_bar)**2)/v_var + ((u - u_bar)**2)/u_var
        t = energy > E_t
        tt =t * 255

        count1 = np.count_nonzero(image_trimap == 255)
        count2 = np.count_nonzero(tt == 255)
        count_rat = count1 / count2
        if 0.85 <= count_rat <= 1.15:
            print("Test passed")
        else:
            raise AssertionError("Trimap provided doesn't match input image")


if __name__ == '__main__':
    unittest.main()
