
function comp = compositing (image, alpha, background)

% COMP = Calculates the composite, given the image with Foreground, Alpha
% Matte and New Backgound.
%   input:
%   image - Image where FG & BG is to be seperated.
%   alpha - alpha matte of the image.
%   background - New background where FG will be composited .
%
%   output:
%   Comp = Final Composite

h = size(image, 1);
w = size(image,2);
backg = imresize(background, [h, w]);
comp = image .* (alpha) + backg .* (1-alpha);

end
