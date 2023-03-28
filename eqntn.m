function [F, B, alpha] = eqntn(mn_F, cov_F, mn_B, cov_B, Chan, cov_C, alpha_init, maxIter, minLike)

% SOLVE     Solves for F,B and alpha that maximize the sum of log
%   likelihoods at the given pixel C.
%   input:
%   mn_F - means of foreground clusters (for RGB, of size 3x#Fclusters).
%   cov_F - covariances of foreground clusters (for RGB, of size
%   3x3x#Fclusters).
%   mu_B,Sigma_B - same for background clusters.
%   Chan - pixel under observation.
%   alpha_init - initial value for alpha.
%   maxIter - maximal number of iterations.
%   minLike - minimal change in likelihood between consecutive iterations.
%
%   output:
%   F, B, alpha - estimate of foreground, background and alpha
%   channel (for RGB, each of size 3x1)

% Step 1: Creating a 3x3 identity matrix I and an empty array vals to 
% store estimated foreground, background, and alpha values and their likelihoods.
I = eye(3);
vals = [];

% Step 2: Loops through each foreground and background cluster to estimate
% their respective means and inverse covariance matrices.
for i = 1 : size(mn_F, 2)
    mu_Fi = mn_F(:, i);
    invSigma_Fi = inv(cov_F(:, :, i));
            
    for j = 1 : size(mn_B, 2)
        mubi = mn_B(:, j);
        invSigmabi = inv(cov_B(:, :, j));
        
        
        % step 3 : Initializing the alpha channel value to alpha_init, 
        % iteration counter to iter, and the last likelihood value to 
        % negative infinity
        alpha = alpha_init;
        iter = 1;
        lastLike = -realmax;

        % Step 4 : Begining a loop that will continue until the maximum 
        % number of iterations has been reached or the change in likelihood
        % between consecutive iterations is below a certain threshold.
        while (true)
            
            % Step 5 : Estimating the foreground and background values 
            % using the current alpha value and the given input values. 
            % Solving for F and B by finding the least squares solution to 
            % the linear system A*X = b, where A is a 6x6 matrix, b is a 
            % 6x1 vector, and X is a 6x1 vector containing F and B. The 
            % values of F and B are then clamped to be between 0 and 1.
            A = [invSigma_Fi + I * (alpha ^ 2 / cov_C ^ 2), I * alpha * (1 - alpha) / cov_C ^ 2; 
               I * ((alpha * (1 - alpha)) / cov_C ^ 2), invSigmabi + I * (1 - alpha) ^ 2 / cov_C ^ 2];
             
            b = [invSigma_Fi * mu_Fi + Chan * (alpha / cov_C ^ 2); 
               invSigmabi * mubi + Chan * ((1 - alpha) / cov_C ^ 2)];
           
            X = A \ b;
            F = max(0, min(1, X(1 : 3)));
            B = max(0, min(1, X(4 : 6)));
            
            % step 6 : Estimating the alpha channel value using the current
            % F and B values by solving for alpha using a formula based on 
            % the pixel value Chan, the foreground value F, and the 
            % background value B.
            alpha = max(0, min(1, ((Chan - B)' * (F - B)) / sum((F - B).^ 2)));
            
            % Step 7 : The likelihood is calculated for the observation (C)
            % as well as for the foreground (F) and background (B)
            % estimates. The likelihood for the observation is calculated 
            % using the sum of squared differences between the observed 
            % pixel and the weighted combination of foreground and 
            % background pixels, where alpha is the weight of the 
            % foreground pixels and (1-alpha) is the weight of the 
            % background pixels. The calculation is normalized by the 
            % covariance cov_C.
            L_C = -sum((Chan - alpha * F - (1 - alpha) * B).^ 2)/ cov_C;
            L_F = -((F - mu_Fi)' * invSigma_Fi * (F - mu_Fi))/2;
            L_B = -((B - mubi)' *invSigmabi * (B - mubi))/2;
            like = L_C + L_F + L_B;
            
            % Step 8 : The code then checks if the maximum number of 
            % iterations (maxIter) has been reached or if the change in 
            % likelihood from the previous iteration is smaller than a 
            % specified threshold (minLike). If either condition is met, 
            % the loop is terminated.

            if iter >= maxIter || abs(like - lastLike) <= minLike
                break;
            end
            
            lastLike = like;
            iter = iter + 1;
        end
        
        % Step 9 :we need only keep the maximal value, but for now we keep all
        % values anyway as there are not many, and we can investigate them
        % later on. If the loop is terminated before reaching the maximum
        % number of iterations or the specified minimum change in 
        % likelihood, the maximum likelihood estimates of F, B, and alpha 
        % are stored in a struct (val) and added to an array (vals) that 
        % keeps track of all estimated values.
        val.F = F;
        val.B = B;
        val.alpha = alpha;
        val.like = like;
        vals=[vals val];
    end
end

% Step 10: return maximum likelihood estimations. Once all possible 
% combinations of foreground and background clusters have been tested, the 
% values of F, B, and alpha that correspond to the highest likelihood 
% estimate (ind) are returned as the final outputs.
[t, ind] = max([vals.like]);
F = vals(ind).F;
B = vals(ind).B;
alpha = vals(ind).alpha;