import pathlib
import cv2
import numpy as np
import albumentations as A
import torch
import yaml
import random
import os
import time
from utils.fileop import File

class Datapath():
    def __init__(self,configpath=r'config\config.yaml'):
        config_path = configpath
        with open(config_path,'r') as f:
            self.config = yaml.load(f,yaml.FullLoader)
        with open(self.config['path'],'r') as f:
            path = yaml.load(f,yaml.FullLoader)
        self.image_path = path['train_path']
        self.label_path = path['train_label']
        self.batch_size = self.config['batch_size']
        self.setfilelist()
    def tovalidate(self):
        with open(self.config['path'],'r') as f:
            path = yaml.load(f,yaml.FullLoader)
        self.image_path = path['val_path']
        self.label_path = path['val_label']
        self.setfilelist()
    def setfilelist(self,*args):
        if len(args) == 1:
            self.filelist = args[0]
        else:
            self.filelist = os.listdir(self.label_path)
            for i,fname in enumerate(self.filelist):
                path = '/'.join([self.image_path,fname])
                if os.path.exists(path):
                    continue
                else:
                    self.filelist.remove(fname)
            if len(self.filelist)==0:
                raise FileExistsError('check whether file and its parent path is existed')

    def getcurrentpath(self, item):
        stop_sign = False
        ind = item * self.batch_size
        if ind >= len(self.filelist):
            stop_sign = True
            return [], [], stop_sign
        img_batch = []
        label_batch = []
        for i in range(self.batch_size):
            if ind >= len(self.filelist):
                ind = 0
            filename = self.filelist[ind]
            name = filename.split('.')[0]
            imgname = name+'.jpg' if os.path.exists('/'.join([self.image_path, name+'.jpg'])) else name+'.png'
            img_path = '/'.join([self.image_path, imgname])
            if not os.path.exists(img_path):
                raise UserWarning(img_path+' not found')
            label_path = '/'.join([self.label_path, filename])
            img_batch.append(img_path)
            label_batch.append(label_path)
            ind += 1
        return img_batch, label_batch, stop_sign

class Dataloader(Datapath):
    def __init__(self,configpath=r'config\config.yaml'):
        super(Dataloader, self).__init__(configpath)
        with open(self.config['class_dict']) as f:
            self.class_dict = yaml.load(f,yaml.FullLoader)
        self.transform = A.Compose([
            A.RandomSizedCrop(min_max_height=(256, 320), height=320, width=480, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.HueSaturationValue(p=0.5),
                A.RGBShift(p=0.7)
            ], p=1),
            A.RandomBrightnessContrast(p=0.5)
        ],
            keypoint_params=A.KeypointParams(format='xy'),
        )

    def __getitem__(self, item):
        calangle = lambda x: -np.arctan((x[1][1] - x[0][1]) / (x[1][0] - x[0][0])) if x[0][0] != x[1][0] else np.pi / 2
        callength = lambda x: np.sqrt((x[1][1] - x[0][1])**2 + (x[1][0] - x[0][0])**2)
        l_cut = self.config['length']
        imgpathlist,labelpathlist,stopsign = self.getcurrentpath(item)
        if stopsign:
            raise StopIteration
        y11, y12, y13, y14 = [], [], [], []
        y21, y22, y23, y24 = [], [], [], []
        heapmap = []
        img_input = []
        for i,path in enumerate(imgpathlist):
            img = self.getimage(imgpathlist,i)
            keypoints = self.getlabel(labelpathlist,i)
            n_img,n_keypoints = self.enhenced(img,keypoints)
            img_input.append(torch.from_numpy(img.transpose([2,0,1])).unsqueeze(0))
            y11_np,y12_np,y13_np,y14_np=np.zeros([20,30]),np.zeros([20,30,3]),np.zeros([20,30,6]),np.zeros([20,30,2])
            y21_np, y22_np, y23_np, y24_np = np.zeros([40, 60]),np.zeros([40, 60,3]),np.zeros([40, 60,6]),np.zeros([40, 60,2])
            heapmap_np = np.zeros([160,240,5])
            for j,obj_kps in enumerate(n_keypoints):
                long = callength([obj_kps[1],obj_kps[2]])
                if long>l_cut:
                    x_ind,y_ind = int(obj_kps[1][1] / self.h * 20),int(obj_kps[1][0] / self.w * 30)
                    # 长度
                    lrelated0 = self.limit(long / (self.config['long_class'][1]),0.5,2)
                    lrelated1 = self.limit(callength([obj_kps[0],obj_kps[1]]) / long,0.5,2)
                    lrelated2 = self.limit(callength([obj_kps[2], obj_kps[3]]) / long,0.5,2)
                    # 中心点
                    x_related = (obj_kps[1][1] - int(x_ind * 16)) / 16
                    y_related = (obj_kps[1][0] - int(y_ind * 16)) / 16
                    # 角度
                    angle = calangle([obj_kps[1],obj_kps[2]])
                    ang = self.config['angle']
                    ang_c = np.argmin(np.abs(np.array([angle-ang[0],angle-ang[1],angle-ang[2]])))
                    angrelated0 = angle / ang[ang_c]
                    angrelated1 = self.limit(calangle([obj_kps[0], obj_kps[1]]) / angle,0.5,1.5)
                    angrelated2 = self.limit(calangle([obj_kps[2], obj_kps[3]]) / angle,0.5,1.5)
                    y11_np[x_ind,y_ind] = 1
                    y12_np[x_ind,y_ind,ang_c] = 1
                    y13_np[x_ind,y_ind] = [lrelated0,lrelated1,lrelated2,angrelated0,angrelated1,angrelated2]
                    y14_np[x_ind,y_ind] = [x_related,y_related]

                else:
                    x_ind, y_ind = int(obj_kps[1][1] / self.h * 40), int(obj_kps[1][0] / self.w * 60)
                    # 长度
                    lrelated0 = self.limit(long / (self.config['long_class'][0]), 0.5, 2)
                    lrelated1 = self.limit(callength([obj_kps[0], obj_kps[1]]) / long, 0.5, 2)
                    lrelated2 = self.limit(callength([obj_kps[2], obj_kps[3]]) / long, 0.5, 2)
                    # 中心点
                    x_related = (obj_kps[1][1] - int(x_ind * 8)) / 8
                    y_related = (obj_kps[1][0] - int(y_ind * 8)) / 8
                    # 角度
                    angle = calangle([obj_kps[1], obj_kps[2]])
                    ang = self.config['angle']
                    ang_c = np.argmin(np.abs(np.array([angle - ang[0], angle - ang[1], angle - ang[2]])))
                    angrelated0 = angle / ang[ang_c]
                    angrelated1 = self.limit(calangle([obj_kps[0], obj_kps[1]]) / angle, 0.5, 1.5)
                    angrelated2 = self.limit(calangle([obj_kps[2], obj_kps[3]]) / angle, 0.5, 1.5)
                    y21_np[x_ind, y_ind] = 1
                    y22_np[x_ind, y_ind, ang_c] = 1
                    y23_np[x_ind, y_ind] = [lrelated0, lrelated1, lrelated2, angrelated0, angrelated1, angrelated2]
                    y24_np[x_ind, y_ind] = [x_related, y_related]
                for k,point in enumerate(obj_kps):
                    heapmap_np[int(point[1]/2)-1:int(point[1]/2)+2,int(point[0]/2)-1:int(point[0]/2)+2,k+1]=1
            y11_tensor = torch.from_numpy(y11_np).unsqueeze(0).unsqueeze(0)
            y12_tensor, y13_tensor, y14_tensor = torch.from_numpy(y12_np.transpose([2, 0, 1])).unsqueeze(0), \
                                                 torch.from_numpy(y13_np.transpose([2, 0, 1])).unsqueeze(0), \
                                                 torch.from_numpy(y14_np.transpose([2, 0, 1])).unsqueeze(0)
            y11.append(y11_tensor.to(torch.float))
            y12.append(y12_tensor.to(torch.float))
            y13.append(y13_tensor.to(torch.float))
            y14.append(y14_tensor.to(torch.float))
            y21_tensor = torch.from_numpy(y21_np).unsqueeze(0).unsqueeze(0)
            y22_tensor, y23_tensor, y24_tensor = torch.from_numpy(y22_np.transpose([2, 0, 1])).unsqueeze(0), \
                                                 torch.from_numpy(y23_np.transpose([2, 0, 1])).unsqueeze(0), \
                                                 torch.from_numpy(y24_np.transpose([2, 0, 1])).unsqueeze(0)
            y21.append(y21_tensor.to(torch.float))
            y22.append(y22_tensor.to(torch.float))
            y23.append(y23_tensor.to(torch.float))
            y24.append(y24_tensor.to(torch.float))
            heapmap_dim0 = 1 - np.sum(heapmap_np,2)
            heapmap_np[:,:,0] = np.where(heapmap_dim0<0,0,1) * heapmap_dim0
            # heapmap_np = cv2.GaussianBlur(heapmap_np,(3,3),15)
            heapmap_tensor = torch.from_numpy(heapmap_np.transpose([2,0,1])).unsqueeze(0)
            heapmap.append(heapmap_tensor)
        x = torch.cat(img_input,0).to(torch.float)
        y1 = [torch.cat(y21,0),torch.cat(y22,0),torch.cat(y23,0),torch.cat(y24,0)]
        y2 = [torch.cat(y11,0),torch.cat(y12,0),torch.cat(y13,0),torch.cat(y14,0)]
        heapmap = torch.cat(heapmap,0)
        return x,(y1,y2,heapmap)

    def __len__(self):
        return len(self.filelist) // self.batch_size + 1

    def getimage(self,plist,item): # plist[item]
        img = cv2.imread(plist[item])
        self.h,self.w = self.config['size_img'][0],self.config['size_img'][1]
        img = cv2.resize(img,[self.w,self.h],interpolation=cv2.INTER_NEAREST)
        return img

    def getlabel(self,plist,item): # plist[item]
        file = File()
        file.read(plist[item])
        label = file.getpointfloat()
        keypoints = []
        for k, v in label.items():
            keypoint = []
            for j, p in enumerate(v):
                x = self.limit(int(p[0] * self.w),0,479)
                y = self.limit(int(p[1] * self.h),0,319)
                keypoint.append((x, y))
            keypoints.append(keypoint)
        return keypoints

    def enhenced(self,image,keypoints):
        t = time.time()
        new_kps = []
        for keypoint in keypoints:
            random.seed(t)
            transformed = self.transform(image=image,keypoints=keypoint)
            if len(transformed['keypoints']) == 4:
                new_kps.append(transformed['keypoints'])
        image_medium = transformed['image']
        label_medium = new_kps
        return image_medium, label_medium

    def limit(self,x,min,max):
        if x<min:
            return min
        elif x>max:
            return max
        else:
            return x

if __name__ == '__main__':
    pass



    # ----------------------测试数据增强方法-----------------------
    # file = File()
    # file.read(r'F:\daylily_w\dataset\label_point\IMGdaylily_00001.txt')
    # label = file.getpointfloat()
    #
    # send_pipe = {}
    # image = cv2.imread('F:\daylily_w\dataset\image\IMGdaylily_00001.jpg')
    # image = cv2.resize(image, [480, 320])
    # i = 1
    # keypoints = []
    # for k, v in label.items():
    #     keypoint = []
    #     for j, p in enumerate(v):
    #         x = int(p[0] * 480)
    #         y = int(p[1] * 320)
    #         keypoint.append((x, y))
    #     keypoints.append(keypoint)
    #     i+=1
    #
    # transform = A.Compose([
    #     A.RandomSizedCrop(min_max_height=(256, 320), height=320, width=480, p=0.5),
    #     A.HorizontalFlip(p=0.5),
    #     A.OneOf([
    #         A.HueSaturationValue(p=0.5),
    #         A.RGBShift(p=0.7)
    #     ], p=1),
    #     A.RandomBrightnessContrast(p=0.5)
    # ],
    #     keypoint_params=A.KeypointParams(format='xy'),
    # )
    #
    #
    # KEYPOINT_COLOR = (0, 255, 0)  # Green
    #
    #
    # def vis_keypoints(image, keypoints, color=KEYPOINT_COLOR, diameter=3):
    #     image = image.copy()
    #
    #     for obj in keypoints:
    #         for (x,y) in obj:
    #             cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)
    #     cv2.imshow('img',image)
    #     cv2.waitKey(0)
    # vis_keypoints(image,keypoints)
    # new_kps = []
    # t = time.time()
    # for keypoint in keypoints:
    #     random.seed(t)
    #     transformed = transform(image=image,keypoints=keypoint)
    #     new_kps.append(transformed['keypoints'])
    # new_img = transformed['image']
    # print(new_kps)
    # vis_keypoints(new_img,new_kps)
