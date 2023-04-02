
import unittest
from PIL import Image, ImageOps
from Baysian_Mat import Bayesian_Matte
import numpy as np
from compositing import compositing
from numpy.testing import assert_array_equal
from Baysian_Mat import get_window
from orchard_bouman_clust import clustFunc


image = np.array(Image.open("C:/Users/aduttagu/Desktop/Bayesian-Matting-Implementation/input_training_lowres/GT06.png"))
image_trimap = np.array(ImageOps.grayscale(Image.open("C:/Users/aduttagu/Desktop/Bayesian-Matting-Implementation/trimap_training_lowres/Trimap1/GT06.png")))
background = np.array(Image.open('C:/Users/aduttagu/Desktop/Bayesian-Matting-Implementation/background.png'))


alpha,pixel_count = Bayesian_Matte(image,image_trimap) 
alpha_disp = alpha * 255
alpha_int8 = np.array(alpha,dtype = int)

comp_Bay = compositing(image, alpha_disp, background)

#unittest Codes
class testcode(unittest.TestCase):
    """ Unit test - 1 : Checks Input Image is correct or not"""
    def test_dimcheck(self):
        # check if it is a 3 channel image
        s3 = np.size(image,2)
        message = "Your input image must be 3 channels"     
        self.assertEqual(s3, 3, message) 


    """ Unit test - 2 : Checks Input and Composite image shape"""
    def test_image_dim(self):
        # Get the shapes of the input image and composite image
        Image1 = image.shape
        Image2 = comp_Bay.shape

        # Assert that the input image and composite image are equal
        self.assertEqual(Image1, Image2, "The shapes of the input image and composite are not equal")


    """ Unit test - 3 : Checks Input image and Alpha size"""
    def test_image_size(self):
        # Get the shapes of input image and alpha
        h1,w1,c1 = image.shape
        h2,w2 = alpha_disp.shape

        # Assert that the sizes are equal
        self.assertEqual(h1, h2, "Height of the input image and alpha matte are not equal")
        self.assertEqual(w1, w2, "Width of the input image and alpha matte are not equal")


    '''Unit test - 4 : Checks the window function is working or not.
    A random 100x100x3 image is created. The get_window() function is called with the center coordinates (25, 25) and the default window size of 75. 
    The expected output is a 75x75x3 sub-image of the input image centered at (25, 25). Again, the assert_array_equal() method is used to compare the 
    shape of the output of the function with the expected shape. '''
    def test_get_window(self):
        m = np.random.rand(100, 100, 3)
        window = get_window(m, 25, 25, 75)
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

if __name__ == '__main__':
    unittest.main()