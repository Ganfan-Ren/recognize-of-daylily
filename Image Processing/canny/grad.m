function img_grad=grad(img)
%img_grad=grad(img)
%计算梯度(单通道)  图片分辨率高低影响运行时间
[m,n]=size(img);
a=[1;2;1];b=[1;0;-1];
sobelx = b*a';
sobely = a*b';
img_grad = struct;
dx = img_conv(img,sobelx);
dy = img_conv(img,sobely);
dx = dx(2:m-2,2:n-2);
dy = dy(2:m-2,2:n-2);
%Mxy = dx.^2+dy.^2;
%Mxy = sqrt(Mxy);
Mxy=abs(dy)+abs(dx);
theta=zeros(m-3,n-3);
for row=1:m-3
    for vol=1:n-3
        if dx(row,vol)~=0
            theta(row,vol) = atan(dy(row,vol)/dx(row,vol))*180/pi;
        end
    end
end
img_grad.dx=dx;
img_grad.dy=dy;
img_grad.Mxy=Mxy;
img_grad.theta = theta;
end