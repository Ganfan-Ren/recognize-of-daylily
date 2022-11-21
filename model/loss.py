import torch.nn as nn
import torch

class Focal_Loss(nn.Module):
    def __init__(self,weight=[1,1,1], num_classes=3, alpha=0.5, gamma=5):
        super(Focal_Loss, self).__init__()
        self.weight = torch.Tensor(weight)
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.crossentropyloss = nn.CrossEntropyLoss(reduction='none')
    def forward(self,inputs,targets):
        c = inputs.shape[1]
        if c <= 1:
            raise UserWarning("Tensor of dim==1 should be > 1")
        det = inputs.transpose(1, 3).contiguous().view(-1, c)
        label = targets.transpose(1, 3).contiguous().view(-1, c)
        logpt = -self.crossentropyloss(det,label)
        pt = torch.exp(logpt)
        loss = -self.alpha * ((1 - pt) ** self.gamma) * logpt
        loss = loss.mean()
        return loss


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.focal_loss = Focal_Loss() # 角度类别
        self.bce = nn.BCELoss() # 置信度
        self.mse = nn.MSELoss()
    def forward(self,det,label):
        y1,y2,y3 = det
        device = y3.device
        l1,l2,l3 = label
        l3 = l3.to(device)
        for i,tensor in enumerate(l1):
            l1[i] = tensor.to(device)
            l2[i] = l2[i].to(device)

        confidobj_loss = self.bce(y1[0],l1[0]) + self.bce(y2[0],l2[0])
        cls_loss = self.focal_loss(y1[1]*l1[0],l1[1]) + self.focal_loss(y2[1]*l2[0],l2[1])
        angleandlength_loss = self.mse(y1[2] * l1[0],l1[2]) + self.mse(y2[2]* l2[0],l2[2])
        center_loss = self.mse(y1[3] * l1[0],l1[3]) + self.mse(y2[3] * l2[0],l2[3])
        heap_loss = self.focal_loss(y3,l3)
        loss = [confidobj_loss,cls_loss,angleandlength_loss,center_loss,heap_loss]
        weights = [0.01,0.5,0.4,0.5,0.05]
        loss_total = torch.Tensor([0]).to(device)
        for i in range(5):
            loss_total = loss_total + weights[i] * loss[i]
        return loss_total

