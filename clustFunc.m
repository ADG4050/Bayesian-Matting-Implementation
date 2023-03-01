% Implementation of color clustering presented by Orchard and Bouman (1991)

%   input:
%   P - pixel measurement
%   w - corresponding weights of pixel calculate using w = a^2 * g.
%   minvar = minimum variance above which the function works

%   Output:
%   mn - cluster means
%   cov - cluster covariances

% The function first creates a structure called "Clus" that contains the 
% input data matrix P and the weights w. It then calls a function called 
% "calc" to calculate the initial mean, covariance, and other statistics 
% of the cluster using the input data and weights.

% The function then initializes a vector called "nodes" with the "Clus" 
% structure. It enters a while loop that continues to split clusters as 
% long as the maximum variance of any cluster in the "nodes" vector is 
% greater than the input "minVar" value. The "split" function is called 
% to split the cluster with the maximum variance into two new clusters, 
% and these clusters are added to the "nodes" vector.

% Finally, the means and covariance matrices for each cluster are 
% extracted from the "nodes" vector and returned as outputs.


function [mn, cov] = clustFunc(P, w, minVar)

    Clus.X = P;
    Clus.w = w;
    Clus = calc(Clus);
    nodes = [Clus];

    while (max([nodes.lambda]) > minVar)
        nodes = split(nodes);
    end

    for i = 1 : length(nodes)
        mn(:, i) = nodes(i).q;
        cov(:, :, i)=nodes(i).R;
    end


    
    function S_clus = calc(S_clus)

        % calculates cluster statistics.  It calculates the weighted mean and 
        % covariance matrix of the input data using the provided weights. 
        % It also calculates the weighted total sum of errors (wtse), 
        % eigenvectors, eigenvalues, and maximum eigenvalue.

        W = sum(S_clus.w);
        % weighted mean
        S_clus.q = sum(repmat(S_clus.w, [1, size(S_clus.X, 2)]).*S_clus.X, 1)/W;
        % weighted covariance
        t = (S_clus.X - repmat(S_clus.q, [size(S_clus.X, 1), 1])).* (repmat(sqrt(S_clus.w), [1, size(S_clus.X, 2)]));
        S_clus.R = (t' * t) / W + 1e-5 * eye(3);

        S_clus.wtse = sum(sum((S_clus.X - repmat(S_clus.q, [size(S_clus.X, 1), 1])).^ 2));

        [V, D] = eig(S_clus.R);
        S_clus.e = V(:, 3);
        S_clus.lambda = D(9);

    
    function nodes = split(nodes)
        
        % splits maximal eigenvalue node in direction of maximal variance. 
        % The "split" function splits the maximal eigenvalue node in the 
        % direction of maximal variance by separating the data points into 
        % two groups based on their projection onto the eigenvector 
        % corresponding to the maximal eigenvalue. It creates two new 
        % clusters, calculates their statistics using the "calc" function,
        % and replaces the original cluster with its two children in the 
        % "nodes" vector.

        [x,i] = max([nodes.lambda]);
        s_node = nodes(i);
        idx = s_node.X * s_node.e <= s_node.q * s_node.e;
        Ca.X = s_node.X(idx, :);
        Ca.w = s_node.w(idx);
        Cb.X = s_node.X(~idx, :); 
        Cb.w = s_node.w(~idx);
        Ca = calc(Ca);
        Cb = calc(Cb);
        nodes(i) = []; % remove the i'th nodes and replace it with its children
        nodes = [nodes, Ca, Cb];