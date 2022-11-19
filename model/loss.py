import torch.nn as nn
import torch
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        weight = torch.Tensor([0.000265,0.974169,0.99341089])
        self.crossentropy = nn.CrossEntropyLoss(weight=weight,reduction='mean')
    # 原来的损失函数，通过循环计算得到
    # def forward(self,det,label):
    #     a = 0
    #     loss = torch.Tensor([0]).to(det.device)
    #     det = det.transpose(1,3).contiguous().view(-1,3)
    #     label = label.transpose(1,3).contiguous().view(-1, 3)
    #     for i in range(det.shape[0]):
    #         loss += self.bce(det[i],label[i]) * 0.1
    #         # if label[i,1]>0 and a==0:
    #         # if self.mse(det[i],label[i]) * 0.1 < 0.01 and a==0:
    #         #     print(det[i],label[i],self.mse(det[i],label[i]) * 0.1)
    #         #     a = 1
    #     return loss

    # 新的损失函数
    def forward(self,det,label):
        det = det.transpose(1, 3).contiguous().view(-1, 3)
        label = label.transpose(1,3).contiguous().view(-1, 3)
        return self.crossentropy(det,label)


class Focal_Loss(nn.Module):
    def __init__(self,num_classes=3, alpha=0.5, gamma=5):
        super(Focal_Loss, self).__init__()
        # weight = [0.000265,0.974169,0.99341089]  # 类别权重
        weight = [0.7,0.3]  # 类别权重
        self.weight = torch.Tensor(weight)

        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.crossentropyloss = nn.CrossEntropyLoss(reduction='none')
    def forward(self,inputs,targets):
        c = inputs.shape[1]
        det = inputs.transpose(1, 3).contiguous().view(-1, c)
        label = targets.transpose(1, 3).contiguous().view(-1, c)
        logpt = -self.crossentropyloss(det,label)
        pt = torch.exp(logpt)
        loss = -self.alpha * ((1 - pt) ** self.gamma) * logpt
        loss = loss.mean()
        return loss



class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

class Loss_2cnet(nn.Module):
    def __init__(self):
        super(Loss_2cnet, self).__init__()
        self.loss1 = Focal_Loss()
        self.loss2 = nn.MSELoss()
    def forward(self,mask1,label1,mask2,label2):
        device = mask1.device
        mask2, label2 = mask2.to(device),label2.to(device)
        mask2_ = torch.ones_like(mask1).to(device)
        label2_ = torch.ones_like(label1).to(device)
        mask2_[:,0,:,:] = mask2_[:,0,:,:] - torch.sum(mask2,1)
        mask2_[:, 1, :, :] = mask2[:, 0, :, :]
        label2_[:, 0] = label2_[:, 0] - label2
        label2_[:, 1] = label2
        loss_mask = self.loss1(mask1,mask2_)
        loss_y = self.loss2(label1,label2_)
        # print('loss_mask: ',float(loss_mask*50),'loss_y: ',float(loss_y))
        return loss_mask*50+loss_y