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
    def getcurrentpath(self,item):
        stop_sign = False
        ind = item * self.batch_size
        if ind >= len(self.filelist):
            stop_sign=True
            return [],[],stop_sign
        img_batch = []
        label_batch = []
        for i in range(self.batch_size):
            if ind >= len(self.filelist):
                ind = 0
            filename = self.filelist[ind]
            img_path = '/'.join([self.image_path,filename])
            label_path = '/'.join([self.label_path, filename])
            img_batch.append(img_path)
            label_batch.append(label_path)
            ind += 1
        return img_batch,label_batch,stop_sign

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
        pass

    def getimage(self,plist,item): # plist[item]
        # img = cv2.imread(plist[item])
        # h, w = self.config['size_img'][0], self.config['size_img'][1]
        # img = cv2.resize(img, [w, h])
        # return img
        #----------------svm_read---------------
        img = cv2.imread(plist[item])
        self.h,self.w = self.config['size_img'][0],self.config['size_img'][1]
        img = cv2.resize(img,[self.h,self.w],interpolation=cv2.INTER_NEAREST)
        return img

    def getlabel(self,plist,item): # plist[item]
        file = File()
        file.read(plist[item])
        label = file.getpointfloat()
        label[:,0] = np.int_(label[:,0]*self.w)
        label[:, 1] = np.int_(label[:, 1] * self.h)
        return label


    def enhenced(self,image,label):
        random.seed(time.time())
        transformed = self.transform(image=image, keypoints=label)
        image_medium = transformed['image']
        mask_medium = transformed['keypoints']
        return image_medium, mask_medium

if __name__ == '__main__':
    file = File()
    file.read(r'F:\daylily_w\dataset\label_point\IMGdaylily_00000.txt')
    label = file.getpointint()
    keypoint = []
    for k,v in label.items():
        for j,p in enumerate(v):
            # x = int(p[0] * 320)
            # y = int(p[1] * 480)
            x = p[0]
            y = p[1]
            keypoint.append((x,y))
    transform = A.Compose([
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
    image = cv2.imread('F:\daylily_w\dataset\image\IMGdaylily_00000.jpg')
    # image = cv2.resize(image,[480,320])
    KEYPOINT_COLOR = (0, 255, 0)  # Green


    def vis_keypoints(image, label, color=KEYPOINT_COLOR, diameter=3):
        image = image.copy()

        for p in keypoint:
            cv2.circle(image, (int(p[0]), int(p[1])), diameter, (0, 255, 0), -1)
        cv2.imshow('img',image)
        cv2.waitKey(0)
    vis_keypoints(image,label)