close all
clear

%fcomp = compositing(comp,comp_s);
%function comp = compositing (comp,comp_s)
org = im2double(imread('input.jpg'));

t_s = im2double(imread('output_alpha.png'));

% Convert to double and make it look a little better
%pic = floor(double(org)*255/(255*255));

back = im2double(imread('background.jpg'));
h= size(org, 1);
w = size(org,2);
back = imresize(back,[h,w]);
%back = double(imread('alpha matte.fig'));
comp1= org .* (t_s) + back .* (~t_s);
figure(7);
image(comp1);


%end
