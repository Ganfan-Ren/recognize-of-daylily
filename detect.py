import torch
import cv2
import numpy as np

def detect(net,img,config):
    pass

def tensor_objabsvalue(det,config): # 将输出转换为[600+2400,9]
    y1,y2,y3 = det
    imgsize = config['size_img']
    long_cls = config['long_class']
    angle_cls = np.array(config['angle'])
    width,height = imgsize[1],imgsize[0]
    # --------------针对y1的计算------------
    size = y1[0].squeeze().shape # [20,30] or [40,60]
    obj_conf = y1[0].reshape(-1,1)
    # ---------------角度------------------
    cls_angle = y1[1].transpose(1, 3).contiguous().view(-1, 3)
    cls_index = torch.argmax(cls_angle,1).view(-1,1)  # [600,1]
    cls_onehot = torch.zeros([cls_index.shape[0],len(angle_cls)]) # [600,3]
    cls_onehot = cls_onehot.scatter(1,cls_index,torch.ones_like(cls_index))
    cls_tensor = torch.from_numpy(angle_cls)
    related_angle = y1[2][:, 3:, :, :].contiguous().view(-1, 3)
    angle0 = related_angle[:,0] + torch.sum(cls_tensor * cls_onehot,1) # [600,1]
    angle1 = angle0 - related_angle[:,1]
    angle2 = related_angle[:,2] - angle0
    # ---------------长度-----------------
    related_length = y1[2][:, :3, :, :].contiguous().view(-1, 3)
    length0 = related_length[:,0] * long_cls[0]
    length1 = related_length[:,1] * length0
    length2 = related_length[:,2] * length0
    # ---------------坐标-----------------
    x_forcal, y_forcal = torch.linspace(0, size[0] - 1, size[0]).view(1, 1, size[0], 1), \
                         torch.linspace(0, size[1] - 1, size[1]).view(1, 1, 1, size[1])
    xy_tensor = torch.cat(
        [height / size[0] * (x_forcal + y1[3][:, 0, :, :]), width / size[1] * (y_forcal + y1[3][:, 1, :, :])], 1)
    xy_tensor = xy_tensor.transpose(1, 3).contiguous().view(-1, 2) # [600,2]
    y1_res = torch.cat([obj_conf,xy_tensor,angle0,angle1,angle2,length0,length1,length2],1)

    # --------------针对y2的计算------------
    size = y2[0].squeeze().shape  # [20,30] or [40,60]
    obj_conf = y2[0].reshape(-1, 1)
    # ---------------角度------------------
    cls_angle = y2[1].transpose(1, 3).contiguous().view(-1, 3) # [2400,3]
    cls_index = torch.argmax(cls_angle, 1).view(-1, 1)  # [2400,1]
    cls_onehot = torch.zeros([cls_index.shape[0], len(angle_cls)])  # [2400,3]
    cls_onehot = cls_onehot.scatter(1, cls_index, torch.ones_like(cls_index))
    cls_tensor = torch.from_numpy(angle_cls) # [3]
    related_angle = y1[2][:, 3:, :, :].contiguous().view(-1, 3) # [2400,3]
    angle0 = related_angle[:, 0] + torch.sum(cls_tensor * cls_onehot, 1)  # [2400,1]
    angle1 = angle0 - related_angle[:, 1]
    angle2 = related_angle[:, 2] - angle0
    # ---------------长度-----------------
    related_length = y1[2][:, :3, :, :].contiguous().view(-1, 3)
    length0 = related_length[:, 0] * long_cls[0]
    length1 = related_length[:, 1] * length0
    length2 = related_length[:, 2] * length0
    # ---------------坐标-----------------
    x_forcal, y_forcal = torch.linspace(0, size[0] - 1, size[0]).view(1, 1, size[0], 1), \
                         torch.linspace(0, size[1] - 1, size[1]).view(1, 1, 1, size[1])
    xy_tensor = torch.cat(
        [height / size[0] * (x_forcal + y1[3][:, 0, :, :]), width / size[1] * (y_forcal + y1[3][:, 1, :, :])], 1)
    xy_tensor = xy_tensor.transpose(1, 3).contiguous().view(-1, 2)  # [600,2]
    y1_res = torch.cat([obj_conf, xy_tensor, angle0, angle1, angle2, length0, length1, length2], 1)


def main():
    pass

if __name__ == '__main__':
    pass
