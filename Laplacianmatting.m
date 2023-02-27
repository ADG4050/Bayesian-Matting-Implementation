
function alpha = Laplacianmatting (img, trimap)

% Laplace matting first uses user-supplied foreground and background 
% markers (trimap) to determine the set of foreground and background 
% pixels in an image, and then uses a Laplace matrix to encode the 
% relationship between these pixels into a system of linear equations.
% The solution of this system of equations represents the opacity of the 
% foreground and background pixels, which can be used to generate an alpha 
% mask to achieve image matting.


% Step 1: Labeling Foreground and Background Pixels The user provides foreground 
% and background labels to determine which pixels belong to the foreground and
% which pixels belong to the background. These markers are usually done by manually 
% drawing foreground and background boundaries. That is the trimap.png image.

% Step 2: Construct the Laplace matrix After labeling the foreground and background pixels, 
% construct the Laplace matrix reflecting the difference between each pixel and its neighbors 
% in the image. The place where the "difference" is large is likely to be the junction of the 
% foreground and the background.

% Step 3: Solve the system of linear equations and find alpha Solve the system of linear 
% equations using the Laplace matrix and foreground/background markers to determine the alpha 
% value of each pixel, indicating the opacity of each pixel in the final matte. Finally, the 
% alpha is "covered" on the original image to complete the cutout.


% The input image is a color image in rgb format; the trimap is a grayscale image
% Load the input image and trimap
%function [img,trimap] = laplace(img,trimap)
% Convert the image and trimap to double precision and normalize
img = im2double(img); % Raw read in uint8 image data to double data (integer to floating point)
trimap = im2double(trimap);             

% Get the size of the image
[m, n, c] = size(img); % m - number of pixels in the x direction, n - number of pixels in the y direction, c - rgb tricolour, number of channels is 3    

% Calculate the foreground, background, and unknown pixels
% The thresholds are divided into foreground, background, and to be confirmed. 
% The parts with a grey level greater than a certain threshold (i.e. very "white") 
% are judged to be foreground, the parts with a grey level less than a certain threshold 
% (very "black") are judged to be background, and the other parts are to be confirmed. 
% The part to be confirmed is further processed to determine whether it is the foreground or the background
fg = trimap > 0.90;             % fg is a matrix as large as the image, 
                                % with an element of 1 indicating that the point is the foreground
bg = trimap < 0.10;             % In bg, an element of 1 means that the point is background
unk = ~(fg | bg);               % In unk, an element of 1 means neither foreground nor background, to be confirmed

Laplacian = del2(img);          % Calculating the Laplace matrix

% Calculate the alpha matte
alpha = zeros(m, n);            % Initialize alpha matrix

for i = 1 : c                     % Calculating the alpha matrix
    subLaplacian = Laplacian(:, :, i);
    alpha(unk) = alpha(unk) + subLaplacian(unk) .^ 2;
end

alpha = 1 - sqrt(alpha ./ c);
alpha(bg(:, :, 1)) = 0;         % Forced repositioning of sections already identified as background
alpha(fg(:, :, 1)) = 1;        

end