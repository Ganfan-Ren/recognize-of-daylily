function img=img_conv(image,K)
[i_rows,i_vols]=size(image);
[dim,~]=size(K);
k_l=int32(dim/2)-1;
img=zeros(i_rows,i_vols);
image=single(image);
for i=1:i_rows
    for j=1:i_vols
        sum_r=0;
        for i1=-k_l:k_l
            for j1=-k_l:k_l
                x=i+i1;y=j+j1;%图像的坐标
                x1=i1+k_l+1;y1=j1+k_l+1;%卷积核的坐标
                if ~(x<=0||y<=0||x>i_rows||y>i_vols)
                    sum_r=sum_r+image(x,y)*K(x1,y1);
                end
            end
        end
        img(i,j)=sum_r;
    end
end
end