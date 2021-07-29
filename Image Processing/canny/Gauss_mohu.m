function image = Gauss_mohu(img,dim)
%image = Gauss_mohu(img,dim)
%输入图片和卷积核维度 仅dim=3或dim=5
K = cal_K(dim);
[~,~,n]=size(img);
if n==3
    img = Gray_t(img);
end
image = conv_img(img,K,dim);
image = uint8(image);
end
function K = cal_K(dim)
%K = cal_K(dim)
%计算出高斯卷积核 dim = 维度
K = zeros(dim);
if dim==3
    n=16;
elseif dim==5
    n=273;
else
    error '维度只能为3或5'
end
for i=1:dim
    for j=1:dim
        [x,y] = k_zuobiao(i,j,dim);
        x = single(x);y = single(y);
        K(i,j) = single(int32(gauss2D(x,y)*n));
    end
end
K = nomal(K);
end
function [x1,y1]=k_zuobiao(x,y,dim)
    n = int32(dim/2)-1;
    x1 = x-n-1;y1 = y-n-1;
    if x > dim || y > dim
        warning 'index out'
    end
end
function K=gauss2D(x,y)
sigma = 1;
c = 1/(2*pi*sigma^2);
K = c*exp(-(x^2+y^2)/(2*sigma^2));
end
function K_n = nomal(K)
[m,n] = size(K);
K_n = zeros(m,n);
s = sum(sum(K));
for i = 1:m
    for j = 1:n
        K_n(i,j) = K(i,j)*(1/s);
    end
end
end