#Please install the following library:
import numpy as np
import math
import cv2
import unittest

# please change the alpha path to ensure the code runs
path_alpha = r'C:\Users\sunshine\Desktop\code\PythonFiles\computationLab\GT01.png'

# please change the ground turth alpha path to ensure the code runs
path_GTalpha = r'C:\Users\sunshine\Desktop\code\PythonFiles\computationLab\GT01.png'

# please change the inpur image path to ensure the code runs
path_input = r'C:\Users\sunshine\Desktop\code\PythonFiles\computationLab\input.png'

class TestAlpha(unittest.TestCase):
    
    # 1.This is the dimension check
    def test_dimcheck(self):
        # read input image 
        image = cv2.imread(path_input)
        # check if it is a 3 channel image
        s3 = np.size(image,2)
        message = "Your input image must be 3 channels"     
        self.assertEqual(s3, 3, message) 
       
    # 2.This is the data type check   
    def test_typecheck(self):
        # read input image 
        image = cv2.imread(path_input)
        # check if the type is float or int
        datatype= type(image)
        t = (datatype=='int'or'float')
        message="Data type should be int or float"
        self.assertTrue(t,message)
        
        
    # 3.This is the size check 
    def test_sizecheck(self): 
        # read alpha and ground truth
        alpha=cv2.imread(path_alpha)
        GTalpha=cv2.imread(path_GTalpha)
        
        # get  the number of row and column
        row_alpha=np.size(alpha,0)
        row_GT=np.size(GTalpha,0)
        col_alpha=np.size(alpha,1)
        col_GT=np.size(GTalpha,1)
        
        message1="The number of rows must be consistent."
        message2="The number of columns must be consistent."
        self.assertEqual(row_alpha, row_GT,message1)
        self.assertEqual(col_alpha, col_GT,message2)    
        
    
    # 4.This is the value check
    def test_valuecheck(self):
        # read alpha 
        alpha=cv2.imread(path_alpha)
        # check if any value is 0 or 1.
        mark =0
        if alpha.any() == 0|1:
            mark=1   
        message="all values in alpha should only be 0 or 1"                 
        self.assertEqual(mark,1,message)


if __name__=='__main__':
    unittest.main()


        



    




        



    