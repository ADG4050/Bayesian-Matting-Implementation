% finds the orientation of the covariance matrices, and adds the camera
% variance to each axis

% This function "addCamVar" takes in two inputs:

% Sigma - a 3D array of size 3-by-3-by-K, where K is the number of
% covariance matrices.
% sigma_C - a scalar representing the camera variance to be added to each
% axis of the covariance matrices
% The function returns an updated version of the input Sigma array.

% The "addCamVar" function first initializes a new 3D array called "Sigma"
% with the same dimensions as the input array. It then iterates over each 
% 2D slice of the input array, which represents a single covariance matrix,
% and performs the following steps:

% 1. Extracts the current covariance matrix and decomposes it into its 
% singular value decomposition (SVD) using the "svd" function.
% 2. Adds the camera variance to each of the diagonal elements of the 
% resulting diagonal matrix "S".
% 3. Reconstructs the updated covariance matrix by multiplying the original
% left and right singular vectors "U" and "V'" with the updated diagonal 
% matrix "Sp" (which now includes the added camera variance).
% 4. Finally, the updated covariance matrix is stored in the corresponding
% slice of the output "Sigma" array.

function Sigma = addCamVar(Sigma, sigma_C)

    Sigma = zeros(size(Sigma));
    for i = 1:size(Sigma, 3)
        Sigma_i = Sigma(:, :, i);
        [U,S,V] = svd(Sigma_i);
        Sp = S + diag([sigma_C^2, sigma_C^2, sigma_C^2]);
        Sigma(:, :, i)=U*Sp*V';
    end