% 5c22 Computational Method Assignment - Implementation of Bayesian Matting
% Author - Group : Yin Yang Artistic Chaos (Yuning, Abhishek, ChaiJie)

clc;
close all;
clear all;

% Step 1 
% Reading the input image, trimap and GT - alpha matte of that image.

image = imread('C:\Users\aduttagu\Downloads\input_training_lowres\GT19.png');
trimap = imread('C:\Users\aduttagu\Downloads\trimap_training_lowres\Trimap1\GT19.png');
GT = imread("C:\Users\aduttagu\Downloads\gt_training_lowres\GT19.png");

% Step 2
% Plotting the Input image and trimap for analysis.
% figure(1);
% subplot(1, 2, 1);
% imshow(image);
% title('Input Image');
% subplot(1, 2, 2);
% imshow(trimap);
% title('Trimap');

% Step 3
% Converting the images to double format, as all matrix computations are
% not possible in unsigned integer 8 format, also double represents (0-255)
% in scale of (0-1).
im = im2double(image);
trimap = im2double(trimap);
GT = im2double(GT);


% Step 4
% Initializing all parameters required for the complete process.
N = 100; % Neighbourhood window size
sigma = 8; % standard deviation of the Gaussian distribution
% clustering parameter
minN = 10; % minimum weights required for clustering function activation 
min_var = 0.05; % minimal cluster variance in order to stop splitting
% optimization parameters
cov_C = 0.01; % assumed camera covariance
maxIter=  1000; % maximal number of iterations
minLike=  1e-6; % minimal change in likelihood between consecutive iterations

% Step 5
% Implementation of Bayesian Matting for Alpha Matte starts here

% Step 5.1
% creates seperate logical matrix of foreground, backgound and unknown
% region. 
bg_reg = (trimap == 0); % background 
fg_reg = (trimap == 1); % foreground 
unkmask = (~ bg_reg & ~ fg_reg); % unknow region 



% Step 5.2
% Making three dimentional matrix (r, g, b), so that the seperate logical 
% matrices made before are applicable for the foreground and background
% matrices. 
F = im;
B = im;
FG = repmat(~fg_reg, [1, 1, 3]);
BG = repmat(~bg_reg, [1, 1, 3]);


% Step 5.3
% with help of FG and BG, separating foreground and background regions from
% the image
F(FG) = 0; % putting all values except FG = 0
B(BG) = 0; % putting all values except BG = 0

% figure(2);
% subplot(1, 2, 1);
% imshow(F);
% title('Only Foreground');
% subplot(1, 2, 2);
% imshow(B);
% title('Only Background');

% Step 5.4
% Creating the alpha matte matrix with FG = 1, BG = 0 and unknown as NaN
% and calculating the number of unknown region cells as Unknown_sum.
alpha = zeros(size(trimap));
alpha(fg_reg) = 1;
alpha(unkmask) = NaN;
Unknown_sum = sum(unkmask(:));

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
sq = strel('square', 3);

unreg = unkmask; %transferring data to prevent data loss and inappropiate use.
n = 1; % while loop initial n parameter.

% Creating a counter to issue warning, once max iteration has reached and 
% stopping the iterative loop. Condition = max iteration described below.
m_i = 0;
max_li = 20;

% Step 5.7
% Creating while loop with all unknown region cells marked NaN and 
% calculating F,B, alpha using an iterative process. 

while n < Unknown_sum

    % Step 5.7.1 certain pixels are removed from the unknown region of the
    % image to detect the edges of the image.
    unreg = imerode(unreg, sq);
    edge_pixels = ~unreg & unkmask;

    % Step 5.7.2 finding the position of the pixels inside the above
    % unknown boundary region
    [Y, X] = find(edge_pixels);

    % Step 5.7.2* Creating a counter to issue warning, once max iteration
    % has reached and stopping the iterative loop.
    % Condition = max iteration described below.
    max_rep = 0; 

    % Step 5.7.3 Creating a for loop inside the unknown region for
    % iterating along the count of Y that we just computed.
    for i = 1 : length(Y)
        % Step 5.7.3.1 taking the pixel and reshaping its 3 channel value 
        % into a 3 X 1 matrix
        x = X(i); y = Y(i);
        chan = reshape(im(y, x, :), [3, 1]); 

        % Step 5.7.3.2 taking the surrounding pixel values around that 
        % pixel. (Used calcsurr_alpha Function to calaculate surrounding 
        % values)
        pix_unr = calcsurr_alpha(alpha, x, y, N);

        % Step 5.7.3.3 taking the surrounding pixel values around that 
        % pixel in foreground region. (Used calcsurr_alpha Function to 
        % calaculate surrounding values). Then Calculating weights of that
        % pixel = unknown pixel matrix squared X gausian matrix and then is
        % structured as required.
        pix_fg = calcsurr_alpha(F, x, y, N);
        wghts_f = (pix_unr.^ 2).* g;
        pix_fg = reshape(pix_fg, N * N, 3);
        pix_fg = pix_fg(wghts_f > 0, :);
        wghts_f = wghts_f(wghts_f > 0);

        
        % Step 5.7.3.4 taking the surrounding pixel values around that 
        % pixel in background region. (Used calcsurr_alpha Function to 
        % calaculate surrounding values). Then Calculating weights of that
        % pixel = (1 - unknown pixel matrix) squared X gausian matrix and then is
        % structured as required.
        pix_bg = calcsurr_alpha(B, x, y, N);
        wghts_b = ((1 - pix_unr).^ 2).* g;
        pix_bg = reshape(pix_bg, N * N, 3);
        pix_bg = pix_bg(wghts_b > 0, :);
        wghts_b = wghts_b(wghts_b > 0);

      

        % Step 5.7.3.5 Clustering function will be done only weights of F &
        % B are more than 10 or else we go to next iteration without
        % clustering.
        if length(wghts_f) < minN || length(wghts_b) < minN
            max_rep = max_rep + 1; 

        % Step 5.7.3.5* If whole iteration in the for loop runs without
        % getting enough background or foreground values for some time, 
        % means the loop has become infinite as no more suitable window to 
        % calculate F,B and alpha, hence condition to break the loop and display the result
        % accordingly.
            if (max_rep == length(Y))
                m_i = m_i + 1;
                if (m_i == max_li)
                    n = Unknown_sum;
                    disp("warning : Max iteration for applying MAP with " + ...
                        "the user set window has reached");
                end
            end
            continue;
        end
        
        % Step 5.7.3.6 Partitioning the foreground and background pixels 
        % (in a weighted manner with a clustering function) and calculate
        % the mean and variance of the cluster.
        [mean_f, cov_f] = clustFunc(pix_fg, wghts_f, min_var);
        [mean_b, cov_b] = clustFunc(pix_bg, wghts_b, min_var);


        % Step 5.7.3.7* update covariances with camera variance, detailly 
        % explained in the function. (Can be ommited if neccessary)
        cov_F = addCamVar(cov_f, cov_C);
        cov_B = addCamVar(cov_b, cov_C);

                
        % Step 5.7.3.7 - Calculates the initial alpha value mean of the 
        % surrounding pixels. 
        alpha_init = nanmean(pix_unr(:));

        % Step 5.7.3.8 - Calculating the F, B and alpha values based on MAP 
        % and Bayes method.
        [fg, bg, a] = eqntn(mean_f, cov_F, mean_b, cov_b, chan, cov_C, alpha_init, maxIter, minLike);


        % Step 5.7.3.9 - Assigning the F, B and alpha values.
        F(y,x,:) = fg;
        B(y,x,:) = bg;
        alpha(y,x) = a;

        % Step 5.7.3.10 - removing from unkowns for the next iteration. 
        unkmask(y,x) = 0;  
        n = n + 1;
    end
    
end

% Step 6
% Displaying the alpha matte created from the given image and trimap
figure(3);
subplot(1, 3, 1);
imshow(im);
title('Input Image');
subplot(1, 3, 2);
imshow(trimap);
title('Trimap');
subplot(1, 3, 3);
imshow(alpha);
title('Alpha-Matte');


% Step 7
% Comparing the alpha matte with the laplacian inbuilt function generated
% alpha matte. (Visual Comparision)
Lalpha = Laplacianmatting (image, trimap);

figure(4);
subplot(1, 3, 1);
imshow(alpha);
title('Bayesian Matting - Alpha Matte');
subplot(1, 3, 2);
imshow(Lalpha);
title('Laplacian Matting - Alpha Matte');
subplot(1, 3, 3);
imshow(GT);
title('Ground Truth - Alpha Matte');

% Step 8 :
% Calculating the SAD between the Groudn Truth and Alpha Matte's obtained
% from Bayesian and Laplacian Matting.
GT = GT(:, :, 1);

% Step 9 : 
% % Calculating the MSE between the Ground Truth and Alpha Matte's obtained
% from Bayesian and Laplacian Matting.

diff3 = (GT - alpha);
mse3 = mean((diff3(:).^2), "omitnan");

diff4 = (GT - Lalpha);
mse4 = mean((diff4(:).^2), "omitnan");

