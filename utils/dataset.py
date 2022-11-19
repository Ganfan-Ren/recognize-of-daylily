import pathlib
import cv2
import numpy as np
import albumentations as A
from utils.plattSMO import svm_2image,img_2value
import torch
import yaml
import random
import os
import time
from utils.boxgenerate import Roiselect

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

    def __getitem__(self, item):
        img_list,label_list,stop = self.getcurrentpath(item)
        if stop:
            raise StopIteration
        x,y = [], []
        for i in range(self.batch_size):
            img = self.getimage(img_list,i)
            lab = self.getlabel(label_list,i,self.class_dict)
            image,label = self.enhenced(img,lab)
            # 转换通道顺序
            image = np.transpose(image,[2,0,1])
            x.append(torch.from_numpy(image).unsqueeze(0))
            label = np.transpose(label, [2, 0, 1])
            y.append(torch.from_numpy(label).unsqueeze(0))
        return torch.cat(x,1),torch.cat(y,1)

    def getimage(self,plist,item): # plist[item]
        # img = cv2.imread(plist[item])
        # h, w = self.config['size_img'][0], self.config['size_img'][1]
        # img = cv2.resize(img, [w, h])
        # return img
        #----------------svm_read---------------
        img = img_2value(plist[item])
        h,w = self.config['size_img'][0],self.config['size_img'][1]
        img = cv2.resize(img,[w,h],interpolation=cv2.INTER_NEAREST)
        return img

    def getlabel(self,plist,item,class_dict): # plist[item]
        img = cv2.imread(plist[item])
        h, w = self.config['size_img'][0], self.config['size_img'][1]
        img = cv2.resize(img, [w, h])
        label = np.zeros([h,w,len(class_dict)])
        for k,v in class_dict.items():
            label[:,:,k] = np.where((img[:,:,0]==v[0])&(img[:,:,1]==v[1])&(img[:,:,2]==v[2]),1,0)
        return label

    def enhenced(self,image,label):
        aug = A.Compose([
            A.OneOf([
                A.RandomSizedCrop(min_max_height=(250, 320), height=image.shape[0], width=image.shape[1], p=0.5),
                A.PadIfNeeded(min_height=image.shape[0], min_width=image.shape[1], p=0.5)
            ], p=1),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
            ], p=0.8)])

        random.seed(time.time())
        augmented = aug(image=image, mask=label)

        image_medium = augmented['image']
        mask_medium = augmented['mask']
        return image_medium, mask_medium




class Dataloader_2(Dataloader):
    def __init__(self,configpath=r'config\config.yaml'):
        super(Dataloader_2, self).__init__(configpath)
        self.h,self.w = self.config['size_img'][0],self.config['size_img'][1]

    def __getitem__(self, item):
        img_list, label_list, stop = self.getcurrentpath(item)
        if stop:
            raise StopIteration
        x = []
        imglist,labellist,anchor = self.get_imglab(img_list,label_list)
        for i,img in enumerate(imglist):
            image = np.transpose(img,[2,0,1])
            x.append(torch.from_numpy(image).unsqueeze(0))
        return torch.cat(x,0),torch.Tensor(np.array(labellist))


    def get_imglab(self,imglist,lablist):
        img,lab,anchor_all = [],[],[]
        for i in range(self.batch_size):
            img_o = self.getimage(imglist,i)
            label_o = self.getlabel(lablist,i,self.class_dict)
            img_,lab_,anchor = self.cutconnect_fromlabel(label_o,img_o)
            img.append(img_)
            lab.append(lab_)
            anchor_all.append(anchor)
        image,label = self.shuffle_spread(img,lab)
        return image,label,anchor_all

    def cutconnect_fromlabel(self,label,image):
        img_,lab_ = [],[]
        anchor_all = {}
        for i in range(1,label.shape[2]):

            imgfor_connect = Roiselect(label[:,:,i],(self.h,self.w))
            imgfor_connect.connect_(1)
            anchor = imgfor_connect.getanchor()

            # label[:, :, 0] = 0
            # label *= 255
            anchor_all[i-1] = []
            for k,v in anchor.items():
                v = np.array(v)
                ima = image[v[0, 1]:v[1, 1], v[0, 0]:v[1, 0], :]
                if ima.shape[0] == 0 or ima.shape[1] == 0:
                    continue
                img_.append(ima)


                # label = cv2.rectangle(np.uint8(label),v[0],v[1],[0,255,255],3)
                # cv2.imshow('label',label)
                # cv2.waitKey(1000)

                lab_.append(i-1)
                anchor_all[i-1].append(v)
        return img_,lab_,anchor_all


    def shuffle_spread(self,img,lab):
        # 随机展开  变形size
        n = 0
        index = []
        img_, lab_ = [], []
        for i,imlist in enumerate(img):
            for j,im in enumerate(imlist):
                try:
                    img_.append(cv2.resize(im,[self.w,self.h]))
                    lab_.append(lab[i][j])
                    index.append(n)
                    n+=1
                except:
                    continue
        return img_,lab_
        # random.shuffle(index)
        # image,label = [],[]
        # for i in range(self.batch_size):
        #     image.append(img_[index[i]])
        #     label.append(lab_[index[i]])
        # return image,label

    def savedataset(self, savepath, item):
        self.n = 0
        img_list, label_list, stop = self.getcurrentpath(item)
        # print(img_list)
        # print(label_list)
        imglist,labellist,anchor = self.get_imglab(img_list,label_list)
        for i,anc in enumerate(anchor):
            name = img_list[i].split('/')[-1].split('.')[0] + '.txt'
            savep = '/'.join([savepath,name])
            with open(savep,'w') as f:
                for k,v in anc.items():
                    for box in v:
                        f.write(str(k)+' ')
                        for point in box:
                            f.write(str(point[0]))
                            f.write(' ')
                            f.write(str(point[1]))
                            f.write(' ')
                        f.write('\n')

class Dataloader_1(Datapath):
    def __init__(self,configpath=r'config\config.yaml'):
        super(Dataloader_1, self).__init__(configpath)
        path = pathlib.Path(self.label_path)
        self.mask_path = self.label_path
        self.label_path = '/'.join([str(path.parent),'image_cut'])
        self.setfilelist()
        self.n = 0  # file  index
        self.c = 0  # class index
        self.box_ind = 0  # box index
        self.h = self.config['size_img'][0]
        self.w = self.config['size_img'][1]


    def readboxfile(self,path):
        if path.split('.')[-1] != 'txt':
            path = path.split('.')[0] + '.txt'
        with open(path, 'r') as f:
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
            y = np.reshape(y, [-1, 2])
            try:
                anchor[int(x)].append(y)
            except:
                anchor[int(x)] = [y]
        return anchor

    def __next__(self):
        img,mask,label = [],[],[]
        boxes = self.readboxfile(self.getcurrentpath(self.label_path,self.filelist,self.n))
        i = 0
        while i < self.batch_size:
            try:
                if self.box_ind >= len(boxes[self.c]):
                    self.box_ind = 0
                    self.c += 1
            except KeyError:
                self.box_ind = 0
                self.c += 1
            if self.c >= len(boxes):
                self.c = 0
                self.box_ind = 0
                self.n += 1
                boxes = self.readboxfile(self.getcurrentpath(self.label_path, self.filelist, self.n))
            if self.n >= len(self.filelist):
                self.n = 0
                self.c = 0  # class index
                self.box_ind = 0  # box index
                raise StopIteration
            try:
                box = boxes[self.c][self.box_ind]
            except KeyError:
                self.box_ind += 1
                continue
            image_ = cv2.resize(cv2.imread(self.getcurrentpath(self.image_path, self.filelist, self.n)),[self.w,self.h])
            mask_l = cv2.resize(cv2.imread(self.getcurrentpath(self.mask_path, self.filelist, self.n)), [self.w, self.h])
            img.append(cv2.resize(self.cutimage(image_,box),[self.w, self.h]).transpose([2,0,1]))
            mask.append(cv2.resize(np.where(self.cutimage(mask_l[:,:,self.c+1], box)==255,1,0),[self.w, self.h],interpolation=cv2.INTER_NEAREST))
            label.append(self.c)
            self.box_ind += 1
            i += 1
        return self.getTensor(img,mask,label)

    def __iter__(self):
        return self



    def getTensor(self,img,mask,label):
        image,masks = [],[]
        labels = torch.from_numpy(np.array(label))
        for i,im in enumerate(img):
            image.append(torch.from_numpy(im).unsqueeze(0))
            masks.append(torch.from_numpy(mask[i]).unsqueeze(0).unsqueeze(0))
        return torch.cat(image,0).to(torch.float),torch.cat(masks,0),labels

    def getcurrentpath(self,path,filelist,index):
        try:
            p = pathlib.Path('/'.join([path,filelist[index]]))
        except:
            index = index - 1
            p = pathlib.Path('/'.join([path,filelist[index]]))
        if os.path.exists(str(p)):
            return str(p)
        else:
            return (str('/'.join([str(p.parent),p.stem])+'.png'))

    def cutimage(self,img,box):
        if len(img.shape) ==3:
            return img[box[0,1]:box[1,1],box[0,0]:box[1,0],:]
        elif len(img.shape) ==2:
            return img[box[0, 1]:box[1, 1], box[0, 0]:box[1, 0]]

if __name__ == '__main__':
    d = Dataloader_2(r'F:\PycharmProject\sweetdetect\config\config.yaml')
    d.savedataset(r'F:\dataset\image_cut',0)