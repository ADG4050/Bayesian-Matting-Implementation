
## 5C22 Computatational Method Assigment - Implementation of Bayesian Matting
Author - Group : Yin Yang Artistic Chaos (Yuning, Abhishek, ChaiJie)

---

## Table of Contents

1. [High-level Description of the project]
2. [Installation and Execution
3. [Methodology and Results]
4. [Credits]

---
## High-level Description of the project
This project we implemented Bayesian matting method. We assume that we have successfully created an alpha matte and we compared the quality of our alpha matte with another alpha matte which created by Laplacian matting method.

--Bayesian matting: We create three matrix in trimap image namely forground matrix, background matrix and unknown region matrix. Then we make a window in the boundary of unknown region. And we also make window in the same region of forground and background. We sum the number of background or forground, if the value more than 10, we will calculate the mean value and variance. Then we calculate the probability of alpha. After we know alpha value we calculate the forground and background value. If the window size is less than 10, we increase window size and repeat the steps. 

--Lapacian matting: We use the trimap image to identify which pixels are foreground and which are background. The Laplacian matrix is constructed to reflect the difference between the marked pixel and its neighboring pixels in the image. Where the "difference" is large is likely to be an area of uncertain pixel values. Then solve the system of linear equations and find the alpha to determine the alpha value for each pixel, resulting in the alpha matrix.

---

## Installation and Execution

1) The pipreqs $project/path/requirements.txt file available in Github provides all the versions and libraries of the import files used in the project. The imports can be installed on a differet machine using 'pip install import name'.

For more details check [here](https://github.com/bndr/pipreqs)
    
    i) numpy==1.23.4
    ii) cv2== 4.7.0
    iii) matplotlib==3.6.2
    iv) scipy.ndimage==1.10.0
    v) PIL==9.4.0
    vi) orchard_bouman_clust==
    vii) numba==0.56.4

2) Afer installing all the import packages, the main file and Bayesian_matting file can be run using the following command in the CMD terminal 
```sh
i) python Main.py
ii) python Bayesian_matting.ipynb

```
---
## Methodology and Results

**Unittests**

Four unittests are checked in this Bayesian matting project to confirm their functionality.

1) Dimension check: Check matrix dimension of input image.
2) Size check: Check size of input image and compositing image.
3) Height and Width check: Check height and weight of input image and alpha matte.
4) Window size check: Check if window size is 75.
5) Orchard Bouman check: Check mean value is [1 * 3] matrix and variance is [3 * 3] matrix.

**e2e tests**

Three e2e tests are checked in this project to get our comparsion results. 
1) Alpha matte comparsion between Bayesian matting and Laplacian matting. 
2) Performance evaluation.
3) Histogram between input images and compositing images.

**Results**

1. Comparing the quality of alpha matte for Bayesian and Laplacian using three evaluation systems namely MSE, SAD and PSNR. A table is made to note the the MSE, SAD and PSNR between the alpha matte and the ground trut obtained by the two matting methods. 
<img src="name.png" width="350">
In order to show the quality difference between the two alpha mattes more intuitively, we take the images GT_01 to GT_10 as examples, and use line graph to show the performance of the three matrices
<img src="name.png" width="350">
The graph shows that alpha matte from Bayesian matting has less MSE(Mean Square Error), less SAD(Sum of Absolute Differences) and large PSNR(peak signal-to-noise ratio). We can draw the conclusion that the quality of alpha matte from Bayesian matting is better.


2. Performance evaluation.
i) Lower size trimap images: Trimap1 hase less unknown region, and trimap2 has more unknown region.
ii) Higher size trimap images


3. Comparing the two histograms between input image and composited image.
Initially our plan was to compare the similarity between the histograms of the input image and the synthesized image. But we forget that the histogram is a sufficient and non-essential condition for the matting result. Therefore, it cannot be explained from the histogram that the result of the matting is correct.

There are two reasons:

i) A histogram of any color space has no positional information. This means that it is entirely possible for two different pictures to have the same histogram.
ii) Unless we use a monochrome image for the background composite, we see a peak and foreground distribution. When the background changes, the peaks move, but the foreground distribution is almost unchanged. But this is not what we want, we want to use a more complex arbitrary background image in the composition. At this time, we cannot accurately distinguish the foreground and background.

---
## Credits

This code was developed for purely academic purposes by XXXX (add github profile name) as part of the module ..... 

Resources:
- XXXX
- XXX





