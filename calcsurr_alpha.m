% This Function calculates and returns the surrounding N-rectangular 
% neighborhood of a matrix m (here alpha), centered at pixel (x,y). 
% Inputs:
% 1) Alpha = The image on which window is to be created
% 2) x = Border x point of the unknown region
% 3) y = Border y point of the unknown region
% 4) window = Neighbourhood size
% Output : window centered around that x and y co-ordinate.

function surr = calcsurr_alpha(alpha, x, y, window)
    [h, w, c] = size(alpha);
    hfW = floor(window/2);
    n1 = hfW;
    n2 = window - hfW - 1;
    surr = nan(window, window, c);
    xmin = max(1, x - n1);
    xmax = min(w, x + n2);
    ymin = max(1, y - n1);
    ymax = min(h, y + n2);
    pxmin = hfW - (x - xmin) + 1; 
    pxmax = hfW + (xmax - x) + 1;
    pymin = hfW - (y - ymin) + 1; 
    pymax = hfW + (ymax - y) + 1;
    surr(pymin : pymax, pxmin : pxmax, :) = alpha(ymin : ymax, xmin : xmax, :);
end