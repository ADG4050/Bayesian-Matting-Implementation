%lab2 2d signal processing
clc; clear; close all;

% Q1.1 2D convolution mask H0
H_0 = [1, 2.5,1; 2.5, 6.25, 2.5; 1, 2.5, 1]/20.25;

%Q1.2  load image into I, convolve I and H0
I = double(imread('kodim19-256.png'))/255;
I0 = conv2(I, H_0,"same");

figure(1);
imshow(I);
figure(2);
imshow(I0);

%Q1.3  1D convolutions mask is H1 and H2
H_1=[1;2.5;1]/4.5;
H_2= [1, 2.5, 1]/4.5;
H_separable= H_1 * H_2;
error_mask= H_separable- H_0;

%Q1.4 use conv2 to obtain the identical image
I0_separable = conv2(I,H_1,"same");
I0_separable = conv2(I0_separable,H_2,"same");

figure(3);
imshow(I0_separable);

%Q1.5 Mean Absolute Error
e_I = I0-I0_separable;
MAE_I0= mean(abs(e_I),"all");

%Q2.1  horizontal derivative Ix.
H_x=[1, 0, -1];
I_x= conv2(I0, H_x,"same");

figure(4);
imshow(I_x+0.5);

%Q2.2  vertical derivative Iy
H_y=[1; 0; -1];
I_y= conv2(I0, H_y,"same");
 
figure(5);
imshow(I_y+0.5);

%Q2.3  image gradient magnitude

grad_mag =I_y^2 + I_x^2;

figure(6);
imshow(grad_mag);

%Q3.1  downsample I by half
% I_downsampling= downsample(I',2);
% I_downsampling= downsample(I_downsampling',2);
% 
% figure(7);
% imshow(I_downsampling);

I_downsampling= imDownSample(I',2);
I_downsampling= imDownSample(I_downsampling',2);

figure(7);
imshow(I_downsampling);


%Q3.2 use h0 to avoid aliasing 

I_downsampling= downsample(I0',2);
I_downsampling= downsample(I_downsampling',2);

figure(8);
imshow(I_downsampling);

%Q4.1  unsharp mask

I_low = I0;
I_high = I -I_low;
Gain= 2.5;
I_sharp= I + Gain* I_high;

figure(9);
imshow(I_sharp);

for i = 3:1:10
    I_sharpest= I + i * I_high;
    figure(i+10);
    imshow(I_sharpest)
end

  













