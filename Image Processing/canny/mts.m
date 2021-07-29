function image=mts(img,x)
%image=mts(img,x)
%Ëõ·ÅÍ¼Æ¬ x=0~1
[m,n]=size(img);
beishu=1/x;
image=zeros(int32(m*x),int32(n*x));
for i=1:int32(m*x)
    for j=1:int32(n*x)
        image(i,j)=img(int32(beishu*i),int32(beishu*j));
    end
end
end