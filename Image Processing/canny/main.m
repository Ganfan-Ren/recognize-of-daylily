img = imread('Å®Éñ±ÚÖ½.jpg');
TH = 200;
TL = 100;
img=Gray_t(img);
img=mts(img,0.5);
imshow(img)
disp('grading...')
img_grad = grad(img);
disp('edgedeting...')
edge = edgedet(img_grad,TL,TH);
imshow(edge)