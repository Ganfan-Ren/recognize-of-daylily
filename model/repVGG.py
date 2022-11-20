import torch
import torch.nn as nn
import torch.nn.functional as F



class Rep(nn.Module):
    def __init__(self,c1,c2,k,s):
        super(Rep, self).__init__()
        padding = int(k/2)
        self.conv1 = nn.Sequential(nn.Conv2d(c1,c2,k,s,padding*2,2),
                                   nn.BatchNorm2d(c2))

        self.conv2 = nn.Sequential(nn.Conv2d(c1,c2,k,s,padding),
                                   nn.BatchNorm2d(c2))

        self.act = nn.ReLU(inplace=True)
    def forward(self,x):
        o1 = self.conv1(x)
        o = o1 + self.conv2(x)
        return self.act(o)

class Rep_2(nn.Module):
    def __init__(self, c1, c2, k, s):
        super(Rep_2, self).__init__()
        padding = int(k / 2)
        self.conv1 = nn.Conv2d(c1, c2, k, s, padding * 2, 2)
        self.conv2 = nn.Conv2d(c1, c2, k, s, padding)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        o1 = self.conv1(x)
        o2 = self.conv2(x)
        o = self.bn(x) + o1 + o2
        return self.act(o)


class RVB1_n(nn.Module):
    def __init__(self,c1,c2,n):
        super(RVB1_n, self).__init__()
        self.conv = nn.Sequential(Rep(c1,c2,3,2),
                                  Rep_2(c2,c2,3,1))
        self.conv2 = Rep_2(c2,c2,3,1)
        self.n = n - 1

    def forward(self,x):
        o = self.conv(x)
        if self.n < 1:
            return o
        for i in range(self.n):
            o = self.conv2(o)
        return o

class SPPF(nn.Module):
    def __init__(self,channel):
        super(SPPF, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(channel,channel,3,1,1),
                                  nn.BatchNorm2d(channel),
                                  nn.ReLU(inplace=True))
        self.maxpool1 = nn.MaxPool2d(3,1,1)
        self.maxpool2 = nn.MaxPool2d(5,1,2)
        self.maxpool3 = nn.MaxPool2d(7,1,3)
        self.conv2 = nn.Sequential(nn.Conv2d(4 * channel,channel,3,1,1),
                                  nn.BatchNorm2d(channel),
                                  nn.ReLU(inplace=True))
    def forward(self,x):
        o1 = self.conv(x)
        o2 = self.maxpool1(x)
        o3 = self.maxpool2(x)
        o4 = self.maxpool3(x)
        o = self.conv2(torch.cat([o1,o2,o3,o4],1))
        return o

class Conv(nn.Module):
    def __init__(self,c1,c2,k,s,p):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(c1,c2,k,s,p),
                                  nn.BatchNorm2d(c2),
                                  nn.ReLU(inplace=True))
    def forward(self,x):
        o = self.conv(x)
        return o

class RepVGG(nn.Module):
    def __init__(self,in_channel):
        super(RepVGG, self).__init__()
        self.c1 = Rep(in_channel,32,3,2)  # [1,32,160,240]
        self.c2 = RVB1_n(32,64,2)    # [1,64,80,120]
        self.c3 = RVB1_n(64,128,4)  # [1,128,40,60]
        self.c4 = RVB1_n(128,256,6)  # [1,256,20,30]
        self.c5 = RVB1_n(256,512,2)  # [1,512,10,15]
        self.maxpool = SPPF(512)  # [1,512,10,15]

    def forward(self,x):
        f1 = self.c1(x)
        f2 = self.c2(f1)
        f3 = self.c3(f2)
        f4 = self.c4(f3)
        f5 = self.c5(f4)
        f5 = self.maxpool(f5)
        return [f1,f2,f3,f4,f5]

