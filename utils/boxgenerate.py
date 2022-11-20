import cv2
import numpy as np
import time

class Roiselect():
    def __init__(self,img,size=(320,480)):
        h,w = size
        self.img = cv2.resize(img,[w,h],interpolation=cv2.INTER_NEAREST)
        self.label = np.zeros([h,w])
        self.mem = []  # 连通区域标签
        self.neibor = np.array([[-1,-1],[-1,0],[-1,1],
                                [0,-1],[0,1],
                                [1,-1],[1,0],[1,1]])
        self.n = 0

    def connect_(self,threhold):
        # t = time.time()
        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                if self.img[i,j] < threhold:
                    continue
                class_ = self.class__(i,j)
                if len(class_) == 1:
                    self.label[i,j] = class_[0]
                    continue
                self.label[i, j] = class_[0]
                self.mem.append(class_)
        self.reflash()
        for s in self.mem:
            for i in range(1,len(s)):
                lab = s[i]
                self.label += np.where(self.label == lab,s[0]-lab,0)
        # print('connect_ use time:',time.time()-t)
        return self.label



    def class__(self,x,y):
        class_ = []
        for index in self.neibor:
            x_ = x + index[0]
            y_ = y + index[1]
            if x_ >= self.label.shape[0] or y_ >= self.label.shape[1]\
                or x_ < 0 or y_ < 0:
                continue
            if self.label[x_,y_] != 0:
                class_.append(self.label[x_,y_])

        if len(class_) == 0:
            self.n += 1
            class_.append(self.n)
        class_ = list(set(class_))
        return class_

    def reflash(self): #将self.mem中的类别整合
        if len(self.mem) == 0:
            return
        i = 0
        while i < len(self.mem)-1:
            j = i + 1
            while j < len(self.mem):
                biaoji = False
                for val in self.mem[j]:
                    if val in self.mem[i]:
                        self.mem[i] = list(set(self.mem[i] + self.mem[j]))
                        self.mem.pop(j)
                        biaoji = True
                        break
                if not biaoji:
                    j += 1
            i += 1
        i = 0
        while i < len(self.mem) - 1:
            j = i + 1
            while j < len(self.mem):
                biaoji = False
                for val in self.mem[j]:
                    if val in self.mem[i]:
                        self.mem[i] = list(set(self.mem[i] + self.mem[j]))
                        self.mem.pop(j)
                        biaoji = True
                        break
                if not biaoji:
                    j += 1
            i += 1

    def getanchor(self):
        pix_index = {}
        anchor = {}
        for i in range(self.label.shape[0]):
            for j in range(self.label.shape[1]):
                if self.label[i,j]==0:
                    continue
                try:
                    pix_index[self.label[i,j]].append([i,j])
                except KeyError:
                    pix_index[self.label[i, j]] = [[i,j]]
        for key,val in pix_index.items():
            pixel = np.array(val)
            xmin = int(np.min(pixel[:,1]))
            ymin = int(np.min(pixel[:, 0]))
            xmax = int(np.max(pixel[:, 1]))
            ymax = int(np.max(pixel[:,0]))
            anchor[key] = [[xmin,ymin],[xmax,ymax]]
        return anchor


    def get_boximage(self,*args):
        '''
        :param args: img,anchor or threhold or todo
        :return:img
        '''
        if len(args) == 1:
            self.connect_(args[0])
            label = np.int_(self.label) * 10
            anchor = self.getanchor()
            a = np.zeros([label.shape[0], label.shape[1], 3])
            for i in range(3):
                a[:, :, i] = label
            for k, v in anchor.items():
                a = cv2.rectangle(a, v[0], v[1], [255, 0, 255], 2)
            return a
        elif len(args) == 2:
            a,anchor = args
            for k, v in anchor.items():
                a = cv2.rectangle(a, v[0], v[1], [255, 0, 255], 2)
            return a

if __name__ == '__main__':
    img = cv2.imread(r'F:\dataset\labels\_2016-05-27-10-26-48_5_frame17.png')
    conect_ = Roiselect(img[:,:,1])
    label = conect_.connect_(50)
    label = np.int_(label)*10
    anchor = conect_.getanchor()
    a = np.zeros([label.shape[0],label.shape[1],3])
    for i in range(3):
        a[:,:,i] = label
    for k,v in anchor.items():
        a = cv2.rectangle(a,v[0],v[1],[255,0,255],2)
    cv2.imshow('123', a)
    cv2.waitKey(0)

    print(anchor)