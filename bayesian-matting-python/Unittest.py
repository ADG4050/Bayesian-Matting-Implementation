
import unittest
from PIL import Image, ImageOps
from Baysian_Mat import Bayesian_Matte
import numpy as np
from compositing import compositing


image = np.array(Image.open("C:/Users/aduttagu/Desktop/Bayesian-Matting-Implementation/input_training_lowres/GT06.png"))
image_trimap = np.array(ImageOps.grayscale(Image.open("C:/Users/aduttagu/Desktop/Bayesian-Matting-Implementation/trimap_training_lowres/Trimap1/GT06.png")))
background = np.array(Image.open('C:/Users/aduttagu/Desktop/Bayesian-Matting-Implementation/background.png'))


alpha,pixel_count = Bayesian_Matte(image,image_trimap) 
alpha_disp = alpha * 255
alpha_int8 = np.array(alpha,dtype = int)

comp_Bay = compositing(image, alpha_disp, background)

#unittest Codes
class testcode(unittest.TestCase):
    """ Unit test - 1 : Checks Input and Composite image shape"""
    def test_image_dim(self):
        # Get the shapes of the input image and composite image
        Image1 = image.shape
        Image2 = comp_Bay.shape

        # Assert that the input image and composite image are equal
        self.assertEqual(Image1, Image2, "The shapes of the input image and composite are not equal")


    """ Unit test - 2 : Checks Input image and Alpha size"""
    def test_image_size(self):
        # Get the shapes of input image and alpha
        h1,w1,c1 = image.shape
        h2,w2 = alpha_disp.shape

        # Assert that the sizes are equal
        self.assertEqual(h1, h2, "Height of the input image and alpha matte are not equal")
        self.assertEqual(w1, w2, "Width of the input image and alpha matte are not equal")
        

if __name__ == '__main__':
    unittest.main()