import torch
import torch.nn as nn
from model.repVGG import RepVGG,Conv


class Rep_2(nn.Module):
    def __init__(self,c1,c2,k,s):
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

class RVB2_n(nn.Module):
    def __init__(self,c1,c2,n):
        super(RVB2_n, self).__init__()
        self.conv1 = nn.Conv2d(c1,c2,3,1,1)
        self.conv2 = nn.Conv2d(c1,c2,1,1)
        self.conv = Rep_2(c2,c2,3,1)
        self.bn = nn.BatchNorm2d(c2)
        self.n = n
    def forward(self,x):
        y1 = self.conv1(x)
        y = self.bn(y1 + self.conv2(x))
        for i in range(self.n):
            y = self.conv(y)
        return y


class SegNet(nn.Module):
    def __init__(self,class_num):
        super(SegNet, self).__init__()
        self.backbone = RepVGG(3)
        # self.backbone2 = RepVGG(1)
        self.conv = nn.ModuleList([Conv(512,256,3,1,1),
                                 Conv(256,128,3,1,1),
                                 Conv(128,64,3,1,1),
                                 Conv(32,class_num,3,1,1),
                                 Conv(512,128,3,1,1)])
        self.conv5_4 = nn.Sequential(RVB2_n(512,512,2),
                                     nn.ConvTranspose2d(512,256,2,2))  # [1,256,20,30]
        self.conv4_3 = nn.Sequential(RVB2_n(256,256,2),
                                     nn.ConvTranspose2d(256,128,2,2))  # [1,128,40,60]
        self.conv3_2 = nn.Sequential(RVB2_n(128,128,2),
                                     nn.ConvTranspose2d(128,64,2,2))    # [1,32,80,120]
        self.conv2_1 = nn.Sequential(RVB2_n(64,64,2),
                                     nn.ConvTranspose2d(64,32,2,2))    # [1,32,160,240]
        self.conv1_0 = nn.Sequential(RVB2_n(32,32,2),
                                     nn.ConvTranspose2d(32,32,2,2))  # [1,32,320,480]
        self.softmax1 = nn.Softmax(1)
        self.linear = nn.Sequential(nn.Linear(19200,1000),
                                    nn.Linear(1000,2))
        self.softmax2 = nn.Softmax(0)


    def forward(self,x):
        feat = self.backbone(x)
        mask = self.conv5_4(feat[4])
        mask = self.conv4_3(self.conv[0](torch.cat([mask,feat[3]],1)))
        mask = self.conv3_2(self.conv[1](torch.cat([mask,feat[2]],1)))
        mask = self.conv2_1(self.conv[2](torch.cat([mask,feat[1]],1)))
        mask = self.conv1_0(mask)
        mask = self.conv[3](mask)
        mask = self.softmax1(mask)
        y = self.conv[4](feat[-1])
        y = self.linear(torch.flatten(y,1,3))
        y = self.softmax2(y)
        return mask,y





if __name__ == '__main__':
    x = torch.rand([2,3,320,480])
    net = SegNet(2)
    mask,y = net(x)
    print(mask.shape,y.shape)

    # x = torch.rand([1,512,10,15])
    # net = nn.Sequential(RVB2_n(512,512,2),
    #                     nn.ConvTranspose2d(512,256,2,2))
    # y = net(x)
    # print(y.shape)