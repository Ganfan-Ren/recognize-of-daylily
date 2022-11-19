import time

import cv2
import tqdm
from utils import Dataloader_2,Dataloader_1
import yaml
import numpy as np
import os

def getpathlist(path):
    # 找比较好的数据集图像
    plist = os.listdir(path)
    l = []
    for i,paths in tqdm.tqdm(enumerate(plist)):
        p = '/'.join([path,paths])
        img = cv2.imread(p)
        # print(np.sum(img[:,:,1]))
        if np.sum(img[:,:,1]) > 50000 * 255 or np.sum(img[:,:,2]) > 50000 * 255:
            l.append(paths)
    return l

def readboxfile(path):
    with open(path,'r') as f:
        file = f.read()
    boxes = file.split('\n')[:-1]
    anchor = {}
    for box in boxes:
        x = box.split(' ')[0]
        p = box.split(' ')[1:-1]
        y = []
        for v in p:
            y.append(int(v))
        y = np.array(y)
        y = np.reshape(y,[-1,2])
        try:
            anchor[int(x)].append(y)
        except:
            anchor[int(x)] = [y]
    return anchor

if __name__ == '__main__':
    # d = Dataloader_2(r'F:\PycharmProject\sweetdetect\config\config.yaml')
    # with open('config/path.txt','r') as f:
    #     i = 0
    #     path = []
    #     p = f.read().split('\n')[:-1]
    # d.setfilelist(p)
    # d.savedataset(r'F:\dataset\image_cut', 0)

    # listp = getpathlist(r'F:\dataset\labels')
    # with open('config/path.txt','w') as f:
    #     for path in listp:
    #         f.write(path)
    #         f.write('\n')
    # print(len(listp))

    # anchor = readboxfile(r'F:\dataset\image_cut\bonirob_2016-05-17-11-42-26_14_frame120.txt')
    # label = cv2.imread(r'F:\dataset\labels\bonirob_2016-05-17-11-42-26_14_frame120.png')
    # label = cv2.resize(label,[480,320])
    # c = [[0,255,255],[255,255,0]]
    # for k,boxes in anchor.items():
    #     for box in boxes:
    #         cv2.rectangle(label,box[0],box[1],c[k],3)
    # cv2.imshow('label',label)
    # cv2.waitKey(0)

    # d = Dataloader_2(r'F:\PycharmProject\sweetdetect\config\config.yaml')
    # with open('config/path.txt','r') as f:
    #     i = 0
    #     path = []
    #     p = f.read().split('\n')[:-1]
    # d.setfilelist(p)
    # for i in tqdm.tqdm(range(len(p)//4)):
    #     d.savedataset(r'F:\dataset\image_cut', i)

    d = Dataloader_1(r'F:\PycharmProject\sweetdetect\config\config.yaml')
    with open('config/path.txt', 'r') as f:
        i = 0
        path = []
        p = f.read().split('\n')[:-1]
    d.setfilelist(p)
    image,mask,label = next(d)
    img = mask[1].squeeze().numpy() * 255
    img = np.uint8(img)
    # img = img.transpose([1,2,0])
    cv2.imshow('2',img)
    cv2.waitKey(0)




