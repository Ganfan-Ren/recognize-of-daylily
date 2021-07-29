function edge=edgedet(img_grad,TL,TH)
Mxy=notismax(img_grad.Mxy,img_grad.theta);
[m,n]=size(Mxy);
edge=zeros(m,n);
con_tinue=1;
for i1=1:m
    for j1=1:n
        if Mxy(i1,j1)>=TH
            edge(i1,j1)=255;
        end
    end
end
jindu=0;
while con_tinue
    imshow(edge)
    con_tinue=0;
    jindu=jindu+1;
    disp(jindu)
    for i2=1:m
        for j2=1:n
            if isedge(Mxy,i2,j2,TL)&&edge(i2,j2)==0&&isedge(edge,i2,j2,255)
                edge(i2,j2)=255;
                con_tinue=1;
            end
        end
    end
end
end
function new_Mxy=notismax(Mxy,theta)
%非极大值抑制
size_m=size(Mxy)+[3,3];
new_Mxy = zeros(size_m(1),size_m(2));
for x1=3:size_m(1)-3
    for y1=3:size_m(2)-3
        x2 = x1-1;y2=y1-1;
        A3_3=[Mxy(x2-1,y2-1),Mxy(x2-1,y2),Mxy(x2-1,y2+1);
            Mxy(x2,y2-1),Mxy(x2,y2),Mxy(x2,y2+1);
            Mxy(x2+1,y2-1),Mxy(x2+1,y2),Mxy(x2+1,y2+1)];
        if ismax_grad(A3_3,theta(x2,y2))
            new_Mxy(x1,y1) = Mxy(x2,y2);
        end
    end
end
end
function y=isedge(Mxy,x,y1,TL)
%判断(x,y)8领域内是否有比TL更大的数
y=0;
[m,n]=size(Mxy);
maxnumber=0;
for i=-1:1
    for j=-1:1
        x1=x+i;y1=y1+j;
        if x1<=0||x1>m||y1<=0||y1>n
            midn=0;
        else
            midn=Mxy(x1,y1);
        end
        if maxnumber<midn
            maxnumber=midn;
        end
    end
end
if maxnumber>=TL
    y=1;
end
end
function max_A3_3 = ismax_grad(A3_3,theta_22)
%非极大值比大小
max_A3_3=0;
if theta_22>=-90&&theta_22<-67.5||theta_22>67.5&&theta_22<=90
    if A3_3(2,2)>A3_3(2,1)&&A3_3(2,2)>A3_3(2,3)
        max_A3_3 = 1;
    end
elseif theta_22>=-67.5&&theta_22<-22.5
    if A3_3(2,2)>A3_3(3,1)&&A3_3(2,2)>A3_3(1,3)
        max_A3_3 = 1;
    end
elseif theta_22>=-22.5&&theta_22<22.5
    if A3_3(2,2)>A3_3(1,2)&&A3_3(2,2)>A3_3(3,2)
        max_A3_3 = 1;
    end
elseif theta_22>=22.5&&theta_22<=67.5
    if A3_3(2,2)>A3_3(3,3)&&A3_3(2,2)>A3_3(1,1)
        max_A3_3 = 1;
    end
else
    error ('角度超出范围：theta = %f',theta_22);
end
end