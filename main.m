% 5c22 Computational Method Assignment - Implementation of Bayesian Matting
% Author - Group : Yin Yang Artistic Chaos (Yuning, Abhishek, ChaiJie)

% Step 1 
% Reading the input image and trimap of that image.
image = imread('input.png');
trimap = imread('trimap.png');

% Step 2
% Plotting the Input image and trimap for analysis.
figure(1);
subplot(1, 2, 1);
imshow(image);
title('Input Image');
subplot(1, 2, 2);
imshow(trimap);
title('Trimap');

% Step 3
% Converting the images to double format, as all matrix computations are
% not possible in unsigned integer 8 format, also double represents (0-255)
% in scale of (0-1).
image = im2double(image);
trimap = im2double(trimap);

% Step 4
% Initializing all parameters required for the complete process.
N = 25; % Neighbourhood window size
sigma = 8; % standard deviation of the Gaussian distribution
sigma_C = 0.01;
minN = 10;
min_var = 0.05;

% Step 5
% Implementation of Bayesian Matting for Alpha Matte starts here

% Step 5.1
% creates seperate logical matrix of foreground, backgound and unknown
% region. 
bg_reg = (trimap == 0); % background 
fg_reg = (trimap == 1); % foreground 
un_reg = (~ bg_reg & ~ fg_reg); % unknow region 

% Step 5.2
% Making three dimentional matrix (r, g, b), so that the seperate logical 
% matrices made before are applicable for the foreground and background
% matrices. 
F = image;
B = image;
FG = repmat(~fg_reg, [1, 1, 3]);
BG = repmat(~bg_reg, [1, 1, 3]);

% Step 5.3
% with help of FG and BG, separating foreground and background regions from
% the image
F(FG) = 0; % putting all values except FG = 0
B(BG) = 0; % putting all values except BG = 0
figure(2);
subplot(1, 2, 1);
imshow(F);
title('Only Foreground');
subplot(1, 2, 2);
imshow(B);
title('Only Background');

% Step 5.4
% Creating the alpha matte matrix with FG = 1, BG = 0 and unknown as NaN
% and calculating the number of unknown region cells as Unknown_sum.
alpha = zeros(size(trimap));
alpha(fg_reg) = 1;
alpha(un_reg) = NaN;
Unknown_sum = sum(un_reg(:));

% Step 5.5
% As per the bayesian matting paper, guassian falloff is used for 
% weighting each pixel neighborhood range set earlier.
% This creates a gaussian filter of size N x N, that can be applied to 
% the image, given the image follows normal distribution. 
g = fspecial('gaussian', N, sigma); 
g = g / max(g(:)); %normalizing it.  


% Step 5.6
% square structuring element for eroding the unknown region(s)
% sq is just an 3X3 square matrix of ones, which is used for image eroding
% at borders.
sq = strel('square',3);

unreg = un_reg; %transferring data to prevent data loss and inappropiate use.
n = 1; % while loop initial n parameter.

% Step 5.7
% Creating while loop with all unknown region cells marked NaN and 
% calculating F,B, alpha using an iterative process. 