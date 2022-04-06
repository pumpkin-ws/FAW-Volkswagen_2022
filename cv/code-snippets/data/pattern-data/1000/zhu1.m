clc
clear
close all
image1=(imread('1.bmp'));
image2=(imread('2.bmp'));
image3=(imread('3.bmp'));
image4=(imread('4.bmp'));
image5=(imread('5.bmp'));
image6=(imread('6.bmp'));
image7=(imread('7.bmp'));
image8=(imread('8.bmp'));
figure,imshow(image1);
figure,imshow(image5);


Image_x{1} = image1;
Image_x{2}= image2;
Image_x{3}= image3;
Image_x{4} = image4;

Image_y{1} = image5;
Image_y{2} = image6;
Image_y{3} = image7;
Image_y{4} = image8;


[mod_x cal_x mod_y cal_y] = cal(Image_x, Image_y);
figure,imshow(mod_x);
figure,imshow(cal_x);
figure,imshow(mod_y);
figure,imshow(cal_y);