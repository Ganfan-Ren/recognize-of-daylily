function img=Gray_t(image)
[~,~,z]=size(image);
if z==1
    img=image;
else
    B=single(image(:,:,1));G=single(image(:,:,2));R=single(image(:,:,3));
    img = R*0.299 + G*0.587 + B*0.114;
    img=uint8(img);
end
end