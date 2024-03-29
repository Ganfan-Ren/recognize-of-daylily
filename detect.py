import torch
import cv2
import yaml
import numpy as np
from model import DayHeap
import os
import argparse
from utils import Detres,Dataloader

def detect(net,img,config,threshold):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    img_size = img.shape
    size = config['size_img']
    img = cv2.resize(img,[size[1],size[0]])
    with torch.no_grad():
        inp = torch.from_numpy(img.transpose([2,0,1])).unsqueeze(0).type(torch.float)
        net = net.to(device)
        inp = inp.to(device)
        det = net(inp)
        res_al = tensor_objabsvalue(det,config) # al are angle and length
        detecter = Detres(res_al,config,threshold)
        keypoint = detecter.heap_fix(24)
    keypoint = np.array(keypoint)
    if len(keypoint) == 0:
        return keypoint
    keypoint[:, :, 0] = keypoint[:, :, 0] / size[1] * img_size[1]
    keypoint[:, :, 1] = keypoint[:, :, 1] / size[0] * img_size[0]
    keypoint = np.int_(keypoint)
    return keypoint

def tensor_objabsvalue(det,config): # 将输出转换为[2400+600,9]
    y1,y2,heapmap = det
    imgsize = config['size_img']
    long_cls = config['long_class']
    angle_cls = np.array(config['angle'])
    width,height = imgsize[1],imgsize[0]
    # --------------针对y1的计算------------
    size = y1[0].squeeze().shape # [20,30] or [40,60]
    obj_conf = y1[0].reshape(-1,1)
    # ---------------角度------------------
    cls_angle = y1[1].transpose(1, 3).transpose(1,2).contiguous().view(-1, 3)  # [1*40*60*3]-->[2400,3]
    cls_index = torch.argmax(cls_angle,1).view(-1,1)  # [1*1*40*60] --> [2400,1]
    cls_onehot = torch.zeros([cls_index.shape[0],len(angle_cls)]).to(cls_index.device) # [2400,3]
    cls_temp = torch.ones_like(cls_index).type(torch.float)
    cls_onehot = cls_onehot.scatter(1,cls_index,cls_temp)
    cls_tensor = torch.from_numpy(angle_cls).to(cls_onehot.device)
    related_angle = y1[2][:, 3:, :, :].transpose(1, 3).transpose(1,2).contiguous().view(-1, 3)
    angle0 = related_angle[:,0] + torch.sum(cls_tensor * cls_onehot,1) # [2400,1]
    angle1 = angle0 - related_angle[:,1]
    angle2 = related_angle[:,2] + angle0
    # ---------------长度-----------------
    related_length = y1[2][:, :3, :, :].transpose(1, 3).transpose(1,2).contiguous().view(-1, 3)
    length0 = related_length[:,0] * long_cls[0]
    length1 = related_length[:,1] * length0
    length2 = related_length[:,2] * length0
    # ---------------坐标-----------------
    x_forcal, y_forcal = torch.linspace(0, size[0] - 1, size[0]).view(1, 1, size[0], 1).to(y1[0].device), \
                         torch.linspace(0, size[1] - 1, size[1]).view(1, 1, 1, size[1]).to(y1[0].device)
    xy_tensor = torch.cat(
        [height / size[0] * (x_forcal + y1[3][:, 0, :, :]), width / size[1] * (y_forcal + y1[3][:, 1, :, :])], 1)
    xy_tensor = xy_tensor.transpose(1, 3).transpose(1,2).contiguous().view(-1, 2) # [2400,2]
    y1_res = torch.cat([obj_conf,xy_tensor,angle0.view(-1,1),angle1.view(-1,1),angle2.view(-1,1),length0.view(-1,1),length1.view(-1,1),length2.view(-1,1),cls_index],1)

    # --------------针对y2的计算------------
    size = y2[0].squeeze().shape  # [20,30] or [40,60]
    obj_conf = y2[0].reshape(-1, 1)
    # ---------------角度------------------
    cls_angle = y2[1].transpose(1, 3).transpose(1,2).contiguous().view(-1, 3) # [600,3]
    cls_index = torch.argmax(cls_angle, 1).view(-1, 1)  # [600,1]
    cls_onehot = torch.zeros([cls_index.shape[0], len(angle_cls)]).to(cls_index.device)  # [600,3]
    cls_temp = torch.ones_like(cls_index).type(torch.float)
    cls_onehot = cls_onehot.scatter(1, cls_index, cls_temp)
    cls_tensor = torch.from_numpy(angle_cls).to(cls_onehot.device) # [3]
    related_angle = y2[2][:, 3:, :, :].transpose(1, 3).transpose(1,2).contiguous().view(-1, 3) # [600,3]
    angle0 = related_angle[:, 0] + torch.sum(cls_tensor * cls_onehot, 1)  # [600,1]
    angle1 = angle0 - related_angle[:, 1]
    angle2 = related_angle[:, 2] + angle0
    # ---------------长度-----------------
    related_length = y2[2][:, :3, :, :].transpose(1, 3).transpose(1,2).contiguous().view(-1, 3)
    length0 = related_length[:, 0] * long_cls[1]
    length1 = related_length[:, 1] * length0
    length2 = related_length[:, 2] * length0
    # ---------------坐标-----------------
    x_forcal, y_forcal = torch.linspace(0, size[0] - 1, size[0]).view(1, 1, size[0], 1).to(y2[0].device), \
                         torch.linspace(0, size[1] - 1, size[1]).view(1, 1, 1, size[1]).to(y2[0].device)
    xy_tensor = torch.cat(
        [height / size[0] * (x_forcal + y2[3][:, 0, :, :]), width / size[1] * (y_forcal + y2[3][:, 1, :, :])], 1)
    xy_tensor = xy_tensor.transpose(1, 3).transpose(1,2).contiguous().view(-1, 2)  # [600,2]
    y2_res = torch.cat([obj_conf, xy_tensor, angle0.view(-1,1), angle1.view(-1,1), angle2.view(-1,1), length0.view(-1,1), length1.view(-1,1), length2.view(-1,1),cls_index], 1)
    y = torch.cat([y1_res,y2_res],0)
    index = torch.linspace(0,y.shape[0]-1,y.shape[0]).view(-1,1).to(y.device)
    y = torch.cat([y,index],1)
    return y,heapmap

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='none', help='initial weights path')
    parser.add_argument('--SOA', type=float, default=-1.0, help='NMS parameter')
    parser.add_argument('--T', type=float, default=-1.0, help='location precision parameter')
    parser.add_argument('--epsilon', type=float, default=-1.0, help='accuracy parameter')
    return parser.parse_known_args()[0] if known else parser.parse_args()

def setconfig():
    opt = parse_opt(True)
    with open('config/config.yaml','r') as f:
        config = yaml.load(f,yaml.FullLoader)
    for key,value in vars(opt).items():
        if isinstance(value,str):
            if value != 'none':
                config[key] = value
        elif isinstance(value,float):
            if value >= 0:
                config[key] = value
    return config

def vis_keypoints(image, keypoints, color, diameter=3):
    image = image.copy()
    for obj in keypoints:
        lastpoint = ()
        # ------------------一起画-----------------
        for i, (x, y) in enumerate(obj):
            cv2.circle(image, (int(x), int(y)), diameter, color[i], -1)
            if len(lastpoint) > 0:
                cv2.line(image,lastpoint,(x,y),(0,0,255),diameter)
            lastpoint = (x,y)
        # ---------------先画点，再画线-------------------
        # for i, (x, y) in enumerate(obj):
        #     cv2.circle(image, (int(x), int(y)), diameter, color[i], -1)
        # for i, (x, y) in enumerate(obj):
        #     if len(lastpoint) > 0:
        #         cv2.line(image, lastpoint, (x, y), (0, 0, 255), diameter)
        #     lastpoint = (x, y)
        # ----------------end--------------------------
    d = np.argmax(image.shape)
    size = [960,int(image.shape[1]/image.shape[0]*960)] if d == 0 else [int(1500 * image.shape[0]/image.shape[1]),1500]
    image = cv2.resize(image, [size[1],size[0]])
    cv2.imshow('img', image)
    cv2.waitKey(0)



def main():
    net = DayHeap()
    config = setconfig()
    imgpath = r'D:\dataset\image\IMGdaylily_00100.jpg'
    if not os.path.exists(imgpath):
        raise FileExistsError('not found ' + imgpath)
    img = cv2.imread(imgpath)
    t = torch.load(config['weight_path'])
    net.load_state_dict(t)
    res = detect(net,img,config,0.3)
    KEYPOINT_COLOR = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 0)]  # B G R
    # print(res)
    vis_keypoints(img,res,KEYPOINT_COLOR,9)


def testmain():
    # loader = Dataloader()
    # loader.batch_size = 1
    # x,y = loader[0]
    # img = x.squeeze().numpy().transpose([1,2,0]) / 255
    # res_al = tensor_objabsvalue(y, loader.config)  # al are angle and length
    # detecter = Detres(res_al, loader.config, 0.3)
    # keypoint = detecter.getkeypoint()
    # image = detecter.vis_heapmap(8)
    # img_1 = np.zeros_like(img)
    # for i in range(3):
    #     img_1[:,:,i] = image
    # img = np.concatenate((img,img_1),1)
    # KEYPOINT_COLOR = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 0)]  # B G R
    # vis_keypoints(img,keypoint,KEYPOINT_COLOR,1)

    net = DayHeap()
    config=setconfig()
    imgpath = r'D:\dataset\image\IMGdaylily_00173.jpg'
    if not os.path.exists(imgpath):
        raise FileExistsError('not found ' + imgpath)
    img = cv2.imread(imgpath)
    t = torch.load(config['weight_path'])
    net.load_state_dict(t)
    KEYPOINT_COLOR = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 0)]  # B G R
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    img_size = img.shape
    size = config['size_img']
    img = cv2.resize(img, [size[1], size[0]])
    with torch.no_grad():
        inp = torch.from_numpy(img.transpose([2, 0, 1])).unsqueeze(0).type(torch.float)
        net = net.to(device)
        inp = inp.to(device)
        det = net(inp)
        res_al = tensor_objabsvalue(det, config)  # al are angle and length
        detecter = Detres(res_al, config, 0.3)
        keypoints = detecter.heap_fix(24)
        image = detecter.vis_heapmap(8)

    img_1 = np.zeros_like(img)
    for i in range(3):
        img_1[:,:,i] = np.uint8(image * 255)
    img = np.concatenate((img,img_1),1)
    vis_keypoints(img, keypoints, KEYPOINT_COLOR, 1)

if __name__ == '__main__':
    testmain()
