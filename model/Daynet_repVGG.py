import torch
import torch.nn as nn
from model.repVGG_tiny import RepVGG_tiny,Conv,Rep_2

class ResNet(nn.Module):
    def __init__(self,c):
        super(ResNet, self).__init__()
        self.conv = nn.Sequential(Conv(c,c,3,1,1),
                                   Conv(c,c,3,1,1))
    def forward(self,x):
        o = self.conv(x)
        return x+o

class ResNet_2(nn.Module):
    def __init__(self,c):
        super(ResNet_2, self).__init__()
        self.conv = nn.Sequential(Conv(c,c,1,1,0),
                                   Conv(c,c,3,1,1))
    def forward(self,x):
        o = self.conv(x)
        return x+o

class CBL_n(nn.Module):
    def __init__(self,c,n):
        super(CBL_n, self).__init__()
        self.n = n // 2
        self.res = ResNet(c)
    def forward(self,x):
        o = self.res(x)
        if self.n > 1:
            for i in range(self.n-1):
                o = self.res(o)
        return o

class CBL_n2(nn.Module):
    def __init__(self,c,n):
        super(CBL_n2, self).__init__()
        self.n = n // 2
        self.res = ResNet_2(c)
    def forward(self,x):
        o = self.res(x)
        if self.n > 1:
            for i in range(self.n-1):
                o = self.res(o)
        return o


class RVB2_n(nn.Module):
    def __init__(self,c1,c2,n):
        super(RVB2_n, self).__init__()
        self.conv1 = nn.Conv2d(c1,c2,3,1,1)
        self.conv2 = nn.Conv2d(c1,c2,1,1)
        self.conv = Rep_2(c2,c2,3,1)
        self.bn = nn.BatchNorm2d(c2)
        self.n = n
        self.convdown = Conv(c2,c2,3,2,1)
    def forward(self,x):
        y1 = self.conv1(x)
        y = self.bn(y1 + self.conv2(x))
        for i in range(self.n):
            y = self.conv(y)
        return self.convdown(y)

class Sigmoid_al(nn.Module):
    def __init__(self,k,b):
        super(Sigmoid_al, self).__init__()
        self.k,self.b = k,b
        self.s = nn.Sigmoid()
    def forward(self,x):
        o = self.s(x) * self.k + self.b
        return o

class CCBL(nn.Module):
    def __init__(self,in_c1,in_c2,out_c):
        super(CCBL, self).__init__()
        self.conv = Conv(in_c2 + in_c1, out_c, 3, 1, 1)
        self.cbl_4 = CBL_n(out_c,4)
    def forward(self,x1,x2):
        y = torch.cat([x1,x2],1)
        y = self.cbl_4(self.conv(y))
        return y


class DetectHead(nn.Module):
    def __init__(self,in_channel):
        super(DetectHead, self).__init__()
        self.c1 = nn.Sequential(CBL_n(in_channel,4),
                                Conv(in_channel,1,1,1,0))  # obj_
        self.c2 = nn.Sequential(CBL_n(in_channel, 4),
                                Conv(in_channel, 3, 1, 1, 0))  # class
        self.c3 = nn.Sequential(CBL_n(in_channel, 4),
                                Conv(in_channel, 6, 1, 1, 0)) # angel and length
        self.c4 = nn.Sequential(CBL_n(in_channel, 4),
                                Conv(in_channel, 2, 1, 1, 0)) # related center (x,y)
        self.softmax = nn.Softmax(1)
        self.sigmoid = nn.Sigmoid()
        self.sigmoid1 = Sigmoid_al(1.5,0.5)
        self.sigmoid2 = Sigmoid_al(1,0.5)

    def forward(self,x):
        obj = self.sigmoid(self.c1(x))
        cla = self.softmax(self.c2(x))
        angle_length = self.c3(x)
        center = self.sigmoid(self.c4(x))
        angle_length[:,:3,:,:] = self.sigmoid1(angle_length[:,:3,:,:])
        angle_length[:,3:,:,:] = self.sigmoid2(angle_length[:,3:,:,:])
        return [obj,cla,angle_length,center]

class DayHeap(nn.Module):
    def __init__(self):
        super(DayHeap, self).__init__()
        self.backbone = RepVGG_tiny()
        self.CBL_4 = CBL_n(256,4)
        self.ccbl_1 = CCBL(256,128,128)
        self.up = nn.Upsample(scale_factor=2.0)
        self.rvb_2 = RVB2_n(128,256,2)
        self.ccbl_2 = CCBL(256,256,256)
        self.ccbl_3 = CCBL(256, 128, 128)
        self.ccbl_4 = CCBL(128, 64, 64)
        self.ccbl_5 = CCBL(64, 32, 32)
        self.detheap = nn.Sequential(CBL_n2(32,2),
                                     Conv(32,5,3,1,1),
                                     nn.Softmax(1))
        self.detecthead1 = DetectHead(128)
        self.detecthead2 = DetectHead(256)
    def forward(self,x):
        featmap = self.backbone(x)
        feat0 = self.CBL_4(featmap[3]) # [2,256,20,30]
        y1 = self.up(feat0)            # [1,256,40,60]
        feat1 = self.ccbl_1(y1,featmap[2]) # [1,128,40,60]
        y1 = self.detecthead1(feat1)
        feat2 = self.ccbl_2(self.rvb_2(feat1),feat0) # [2,256,20,30]
        y2 = self.detecthead2(feat2)
        y3 = self.ccbl_3(self.up(feat2),featmap[2]) # [1,128,40,60]
        y3 = self.ccbl_4(self.up(y3),featmap[1]) # [1,64,80,120]
        y3 = self.ccbl_5(self.up(y3), featmap[0]) # [1,32,160,240]
        y3 = self.detheap(y3)
        return y1,y2,y3
        pass

if __name__ == '__main__':
    x = torch.rand([2,3,320,480])
    net = DayHeap()
    y1,y2,y3 = net(x)
    for tensor in y1:
        print(tensor.shape)
    for tensor in y2:
        print(tensor.shape)
    print(y3.shape)

    # x = torch.rand([1,512,10,15])
    # net = nn.Sequential(RVB2_n(512,512,2),
    #                     nn.ConvTranspose2d(512,256,2,2))
    # y = net(x)
    # print(y.shape)