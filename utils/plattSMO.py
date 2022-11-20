import time

import numpy as np
import cv2
import utils.svm as svm
from os import listdir
import pickle
import random
import os

def YBCr_Color(src):
    YCrcb_img = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
    Y = YCrcb_img[:, :, 0]
    Cr = YCrcb_img[:, :, 1]
    B = src[:, :, 0]
    YBCr = np.zeros_like(src)
    YBCr[:, :, 0] = Y
    YBCr[:, :, 1] = B
    YBCr[:, :, 2] = Cr
    return YBCr

def HSI_Color(src):
    HSV_img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    H = HSV_img[:, :, 0]
    S = HSV_img[:, :, 1]
    R = src[:, :, 2]
    G = src[:, :, 1]
    B = src[:, :, 0]
    I = np.int_((0.596 * R) - (0.274 * G) - (0.322 * B))
    HSI = np.zeros_like(src)
    HSI[:, :, 0] = H
    HSI[:, :, 1] = S
    HSI[:, :, 2] = I
    return HSI

def color_channel_(src, color_channel):
    '''
    :param src: image np.ndarray
    :param color_channel: (str) LAB,HSV,YCrCb,YIQ
    :return: BGR conver to color_channel
    '''

    if color_channel == 'LAB':
        img = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    elif color_channel == 'HSV':
        img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    elif color_channel == 'YCrCb':
        img = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
    elif color_channel == 'RGB':
        img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    elif color_channel == 'YIQ':
        w, h, c = src.shape
        R = src[:, :, 2]
        G = src[:, :, 1]
        B = src[:, :, 0]
        Y = np.int_((0.299 * R) + (0.587 * G) + (0.114 * B))
        I = np.int_((0.596 * R) - (0.274 * G) - (0.322 * B))
        Q = np.int_((0.211 * R) - (0.523 * G) + (0.312 * B))
        img = np.zeros([w, h, 3])
        img[:, :, 0] = Y
        img[:, :, 1] = I
        img[:, :, 2] = Q
    else:
        img = src
    return img


class PlattSMO:
    def __init__(self, dataMat, classlabels, C, toler, maxIter, **kernelargs):
        self.x = np.array(dataMat)
        self.label = np.array(classlabels).transpose()
        self.C = C
        self.toler = toler
        self.maxIter = maxIter
        self.m = np.shape(dataMat)[0]  # 数据总量 ：3997
        self.n = np.shape(dataMat)[1]  # 数据维度
        self.alpha = np.array(np.zeros(self.m), dtype='float64')
        self.b = 0.0
        self.eCache = np.array(np.zeros((self.m, 2)))  # 猜测和误差有关（维度是（3997，2））
        self.K = np.zeros((self.m, self.m), dtype='float64')  # （3997，3997）
        self.kwargs = kernelargs  # {dict} ('name':'rbf', 'theta':20)
        self.SV = ()
        self.SVIndex = None
        for i in range(self.m):
            for j in range(self.m):
                self.K[i, j] = self.kernelTrans_(self.x[i, :], self.x[j, :])

    def calcEK(self, k):
        fxk = np.dot(self.alpha * self.label, self.K[:, k]) + self.b
        Ek = fxk - float(self.label[k])
        return Ek

    def updateEK(self, k):
        Ek = self.calcEK(k)

        self.eCache[k] = [1, Ek]

    def selectJ(self, i, Ei):
        maxE = 0.0
        selectJ = 0
        Ej = 0.0
        validECacheList = np.nonzero(self.eCache[:, 0])[0]
        if len(validECacheList) > 1:
            for k in validECacheList:
                if k == i: continue
                Ek = self.calcEK(k)
                deltaE = abs(Ei - Ek)
                if deltaE > maxE:
                    selectJ = k
                    maxE = deltaE
                    Ej = Ek
            return selectJ, Ej
        else:
            selectJ = svm.selectJrand(i, self.m)
            Ej = self.calcEK(selectJ)
            return selectJ, Ej

    def innerL(self, i):
        Ei = self.calcEK(i)
        if (self.label[i] * Ei < -self.toler and self.alpha[i] < self.C) or \
                (self.label[i] * Ei > self.toler and self.alpha[i] > 0):
            self.updateEK(i)
            j, Ej = self.selectJ(i, Ei)
            alphaIOld = self.alpha[i].copy()
            alphaJOld = self.alpha[j].copy()
            if self.label[i] != self.label[j]:
                L = max(0, self.alpha[j] - self.alpha[i])
                H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
            else:
                L = max(0, self.alpha[j] + self.alpha[i] - self.C)
                H = min(self.C, self.alpha[i] + self.alpha[j])
            if L == H:
                return 0
            eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
            if eta >= 0:
                return 0
            self.alpha[j] -= self.label[j] * (Ei - Ej) / eta
            self.alpha[j] = svm.clipAlpha(self.alpha[j], H, L)
            self.updateEK(j)
            if abs(alphaJOld - self.alpha[j]) < 0.00001:
                return 0
            self.alpha[i] += self.label[i] * self.label[j] * (alphaJOld - self.alpha[j])
            self.updateEK(i)
            b1 = self.b - Ei - self.label[i] * self.K[i, i] * (self.alpha[i] - alphaIOld) - \
                 self.label[j] * self.K[i, j] * (self.alpha[j] - alphaJOld)
            b2 = self.b - Ej - self.label[i] * self.K[i, j] * (self.alpha[i] - alphaIOld) - \
                 self.label[j] * self.K[j, j] * (self.alpha[j] - alphaJOld)
            if 0 < self.alpha[i] and self.alpha[i] < self.C:
                self.b = b1
            elif 0 < self.alpha[j] and self.alpha[j] < self.C:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0

    def smoP(self):
        iter = 0
        entrySet = True
        alphaPairChanged = 0
        while iter < self.maxIter and ((alphaPairChanged > 0) or (entrySet)):
            alphaPairChanged = 0
            if entrySet:
                for i in range(self.m):
                    alphaPairChanged += self.innerL(i)
                iter += 1
            else:
                nonBounds = np.nonzero((self.alpha > 0) * (self.alpha < self.C))[0]
                for i in nonBounds:
                    alphaPairChanged += self.innerL(i)
                iter += 1
            if entrySet:
                entrySet = False
            elif alphaPairChanged == 0:
                entrySet = True
        self.SVIndex = np.nonzero(self.alpha)[0]
        self.SV = self.x[self.SVIndex]
        self.SVAlpha = self.alpha[self.SVIndex]
        self.SVLabel = self.label[self.SVIndex]
        self.x = None
        self.K = None
        self.label = None
        self.alpha = None
        self.eCache = None

    #   def K(self,i,j):
    #       return self.x[i,:]*self.x[j,:].T
    def kernelTrans_(self, x, z):
        if np.array(x).ndim != 1 or np.array(x).ndim != 1:
            raise Exception("input vector is not 1 dim")
        if self.kwargs['name'] == 'linear':
            return sum(x * z)
        elif self.kwargs['name'] == 'rbf':
            theta = self.kwargs['theta']
            return np.exp(sum((x - z) * (x - z)) / (-1 * theta ** 2))  # 核函数

    def kernelTrans(self, x, z):
        if np.array(x).ndim != 1 or np.array(x).ndim != 1:
            raise Exception("input vector is not 1 dim")
        if self.kwargs['name'] == 'linear':
            return sum(x * z)
        elif self.kwargs['name'] == 'rbf':
            theta = self.kwargs['theta']
            r = np.exp(np.sum((x - z) * (x - z),1) / (-1 * theta ** 2))
            return r  # 核函数

    def calcw(self):
        for i in range(self.m):
            self.w += np.dot(self.alpha[i] * self.label[i], self.x[i, :])

    # def predict(self, testData):
    #     test = np.array(testData)
    #     # return (test * self.w + self.b).getA()
    #     result = []
    #     m = np.shape(test)[0]
    #     for i in range(m):
    #         tmp = self.b
    #         for j in range(len(self.SVIndex)):
    #             tmp += self.SVAlpha[j] * self.SVLabel[j] * self.kernelTrans(self.SV[j], test[i, :])
    #         while tmp == 0:
    #             tmp = svm.random.uniform(-1, 1)
    #         if tmp > 0:
    #             tmp = 1
    #         else:
    #             tmp = -1
    #         result.append(tmp)
    #     return result

    def predict(self, testData):
        test = np.array(testData)
        # return (test * self.w + self.b).getA()
        result = []
        m = np.shape(test)[0]
        tmp = self.b * np.ones(m)
        for j in range(len(self.SVIndex)):
            tmp += self.SVAlpha[j] * self.SVLabel[j] * self.kernelTrans(self.SV[j], test)
        a = np.where(tmp == 0,svm.random.uniform(-1, 1),0)
        b = np.where(tmp > 0,1,0)
        c = np.where(tmp < 0, -1, 0)
        result = a + b + c
            # tmp = svm.random.uniform(-1, 1)
        return result

    def saveSV(self, path):
        mine = {'C': self.C, 'K': self.K, 'SV': self.SV, 'SVAlpha': self.SVAlpha, 'alpha': self.alpha, 'b': self.b,
                'kwargs': self.kwargs}
        mine['SVIndex'] = self.SVIndex
        mine['SVLabel'] = self.SVLabel
        mine['m'] = self.m
        if path[-4:] != '.pkl':
            path = path + '/SVM.pkl'
        with open(path, 'wb') as f:
            pickle.dump(mine, f)
        print('smo result has saved in path:\'',path,'\'')

    def readpath(self, path):
        with open(path, 'rb') as f:
            mine = pickle.load(f)
        self.C, self.K, self.SV, self.SVAlpha, self.alpha, self.b, self.kwargs = mine['C'], mine['K'], mine['SV'], mine[
            'SVAlpha'], mine['alpha'], mine['b'], mine['kwargs']
        self.SVIndex = mine['SVIndex']
        self.SVLabel = mine['SVLabel']


# def plotBestfit(data, label, w, b):
#     import matplotlib.pyplot as plt
#     n = np.shape(data)[0]
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     x1 = []
#     x2 = []
#     y1 = []
#     y2 = []
#     for i in range(n):
#         if int(label[i]) == 1:
#             x1.append(data[i][0])
#             y1.append(data[i][1])
#         else:
#             x2.append(data[i][0])
#             y2.append(data[i][1])
#     ax.scatter(x1, y1, s=10, c='red', marker='s')
#     ax.scatter(x2, y2, s=10, c='green', marker='s')
#     x = np.arange(-2, 10, 0.1)
#     y = ((-b - w[0] * x) / w[1])
#     plt.plot(x, y)
#     plt.xlabel('X')
#     plt.ylabel('y')
#     plt.show()


# def loadImage(dir, maps=None):
#     dirList = listdir(dir)
#     data = []
#     label = []
#     for file in dirList:
#         label.append(file.split('_')[0])
#         lines = open(dir + '/' + file).readlines()
#         row = len(lines)
#         col = len(lines[0].strip())
#         line = []
#         for i in range(row):
#             for j in range(col):
#                 line.append(float(lines[i][j]))
#         data.append(line)
#         if maps != None:
#             label[-1] = float(maps[label[-1]])
#         else:
#             label[-1] = float(label[-1])
#     return np.array(data), np.array(label)
#
#
def read_sample(path):
    with open(path) as f:
        a = f.read()
        data_ = a.split('\n')[0].split(' ')[:-1]
        label_ = a.split('\n')[1].split(' ')[:-1]
        data, label = np.zeros(len(data_)), np.zeros(len(label_))
        for i in range(len(data_)):
            data[i] = int(data_[i])
        for i in range(len(label_)):
            label[i] = int(label_[i])
        data = data.reshape([-1, 3])
    return data, label

# def svm_image():
#     a = 'RGB'
#     file = 'D:/sweet vagetable/image/img1'+ a + '.jpg'
#     img = cv2.imread(r'D:\sweetvegetable\test\image\bonirob_2016-05-23-10-47-22_2_frame57.png')
#     cv2.imwrite(file,img)
#     size = img.shape[:2]
#     img = cv2.resize(img,[100,100])
#     img = YBCr_Color(img)          # BGR 通道转换 YBCr
#     # img = HSI_Color(img)
#     # 初始化smo
#     data, label = read_sample('D:/sweetvegetable/data_BGR.txt')
#     smo = PlattSMO(data[:1], label[:1], 200, 0.0001, 10000, name='rbf', theta=20)
#     smo.readpath('D:/sweetvegetable/smo_ret.pkl') # HSI通道结果
#     # 开始预测
#     data = img.reshape([img.shape[0]*img.shape[1],3])
#     result = np.array(smo.predict(data))
#     result = result.reshape([100,100])
#     result = (result + 1) * 125
#     result = np.uint8(result)
#     result = cv2.resize(result,[size[1],size[0]])
#     thre, result = cv2.threshold(result, 0, 255, cv2.THRESH_OTSU)
#     # 预测结束
#     # 保存
#     file = 'D:/sweet vagetable/image/result_svm_BGR'+ a + '.jpg'
#     cv2.imwrite(file,result)
#     # 膨胀处理
#     file = 'D:/sweet vagetable/image/result_add_BGR' + a + '.jpg'
#     k = np.ones((5, 5), np.uint8)
#     image = cv2.dilate(result, k)
#     cv2.imwrite(file,image)
    # 显示预测结果
    # cv2.imshow('result',result)
    # cv2.imshow('image',image)
    # cv2.waitKey(0)


# 训练作者原来的数据
# def main():
#     '''
#     data,label = loadDataSet('testSetRBF.txt')
#     smo = PlattSMO(data,label,200,0.0001,10000,name = 'rbf',theta = 1.3)
#     smo.smoP()
#     smo.calcw()
#     print smo.predict(data)
#     '''
#     maps = {'1': 1.0, '9': -1.0}
#     data,label = loadImage("digits/trainingDigits",maps) # 原来的数据
#     smo = PlattSMO(data[:1], label[:1], 200, 0.0001, 10000, name='rbf', theta=20)
#     smo.smoP()
#     print(len(smo.SVIndex))
#     # smo.saveSV('D:/sweetvegetable')
#     test, testLabel = loadImage("digits/testDigits", maps)
#     testResult = smo.predict(test)
#     m = np.shape(test)[0]
#     count = 0.0
#     for i in range(m):
#         if testLabel[i] != testResult[i]:
#             count += 1
#     print("classfied error rate is:", count / m)
#     smo.kernelTrans(data,smo.SV[0])

# 从图像和标签中获取数据集
# def getdata():
#     src_path = 'D:/sweetvegetable/image'
#     lab_path = 'D:/sweetvegetable/labels'
#     imaglist = os.listdir(lab_path)
#     random.shuffle(imaglist)
#     size = [120, 80]
#     data = np.zeros([0, 15])
#     y = np.zeros(0)
#     readtqdm = tqdm.tqdm(enumerate(imaglist[:50]))
#     readtqdm.set_description('读取数据')
#     for i, name in readtqdm:
#         # print(i,'/',len(imaglist)) # 进度
#         label_img = cv2.resize(cv2.imread('/'.join([lab_path, name])), size).reshape(-1, 3)  # 读取标签图片
#         image = cv2.resize(cv2.imread('/'.join([src_path, name])), size)  # BGR
#         image_LAB = color_channel_(image, color_channel='LAB').reshape(-1, 3)
#         image_HSV = color_channel_(image, color_channel='HSV').reshape(-1, 3)
#         image_YCrCb = color_channel_(image, color_channel='YCrCb').reshape(-1, 3)
#         image_YIQ = color_channel_(image, color_channel='YIQ').reshape(-1, 3)
#         image = image.reshape(-1, 3)
#         c1 = np.zeros([0, 15])
#         c2 = np.zeros([0, 15])
#         c3 = np.zeros([0, 15])
#         for j, color in enumerate(label_img):
#             if color[1] > 150:  # 作物
#                 c1 = np.concatenate([c1, np.concatenate(
#                     [image[j], image_LAB[j], image_HSV[j], image_YCrCb[j], image_YIQ[j]], 0).reshape(-1, 15)], 0)
#             if color[2] > 150:  # 杂草
#                 c2 = np.concatenate([c2, np.concatenate(
#                     [image[j], image_LAB[j], image_HSV[j], image_YCrCb[j], image_YIQ[j]], 0).reshape(-1, 15)], 0)
#             else:
#                 c3 = np.concatenate([c3, np.concatenate(
#                     [image[j], image_LAB[j], image_HSV[j], image_YCrCb[j], image_YIQ[j]], 0).reshape(-1, 15)], 0)
#         c_fore = np.concatenate([c1, c2], 0)
#         c_back = c3
#         k1 = random.randint(0, c_back.shape[0] - 1)
#         if c_fore.shape[0] > 0:
#             k2 = random.randint(0, c_fore.shape[0] - 1)
#             data = np.concatenate([data, c_back[k1].reshape(-1, 15), c_fore[k2].reshape(-1, 15)], 0)
#             y = np.append(y, [-1, 1])
#         else:
#             data = np.concatenate([data, c_back[k1].reshape(-1, 15)], 0)
#             y = np.append(y, [-1])
#     return data,y

# 训练支持向量机的程序（自己的数据）
# def main_1():
#     data,label = getdata()
#     data = data[:,[6,7,13]]
#     num = 30 # 样本呢数量
#     smo = PlattSMO(data[:num], label[:num], 200, 0.0001, 10000, name='rbf', theta=20)
#     smo.smoP()
#     smo.saveSV('D:/sweetvegetable1/smo_ret_BGR.pkl')
#     print(len(smo.SVIndex))

# 得到路径中该图像和经过前背景过滤后图像的输入神经网络的图像
def img_2value(path,channel = 'BGR'):
    # 初始化smo
    if channel=='BGR':
        svm_p = r'svm_data/smo_ret_BGR.pkl'
        channel_to = lambda x: x
    elif channel=='YBCr':
        svm_p = r'svm_data/smo_ret.pkl'
        channel_to = YBCr_Color
    elif channel == 'HSI':
        svm_p = r'svm_data/smo_ret_HSI.pkl'
        channel_to = HSI_Color
    else:
        return

    data, label = read_sample('svm_data/data.txt')
    smo = PlattSMO(data[:1], label[:1], 200, 0.0001, 10000, name='rbf', theta=20)
    smo.readpath(svm_p)
    # 读取并转换图像
    img_ = cv2.imread(path)
    size = img_.shape[:2]
    img_1 = channel_to(img_)
    img = cv2.resize(img_1, [100, 100])
    #
    # # 预测
    data = img.reshape([img.shape[0] * img.shape[1], 3])
    result = np.array(smo.predict(data))
    result = result.reshape([100, 100])
    result = (result + 1) * 125
    result = np.uint8(result)
    result = cv2.resize(result, [size[1], size[0]])
    thre, result = cv2.threshold(result, 0, 255, cv2.THRESH_OTSU)
    # # 预测结束
    # # 膨胀处理
    # k = np.ones((5, 5), np.uint8)
    # image = cv2.dilate(result, k)
    # # 原图去除背景
    # # img_1[:,:,2] = image/255 * img_1[:,:,2]
    # # return img_1
    # # 在原图上增加预测结果维度，增加通道数
    shape_ = list(size)
    shape_.append(4)
    img_2 = np.zeros(shape_)
    img_2[:,:,:3] = img_1
    img_2[:,:,3] = result
    return img_2


def svm_2image(path,channel='YBCr'):
    # 选择颜色通道的SVM参数(默认YBCr)
    if channel == 'BGR':
        svm_path = r'svm_data/smo_ret_BGR.pkl'
        channel_to = lambda x: x
    elif channel == 'YBCr':
        svm_path = r'svm_data/smo_ret.pkl'
        channel_to = YBCr_Color
    elif channel == 'HSI':
        svm_path = r'svm_data/smo_ret_HSI.pkl'
        channel_to = HSI_Color
    else:
        return
    # 初始化smo
    data, label = read_sample('svm_data/data.txt')
    smo = PlattSMO(data[:1], label[:1], 200, 0.0001, 10000, name='rbf', theta=20)
    smo.readpath(svm_path)
    # 读取并转换图像
    img_1 = cv2.imread(path)
    # -------------------------------------------BGR分割线
    size = img_1.shape[:2]
    img_1 = channel_to(img_1)
    #--------------------------------------------转换通道分割线
    img = cv2.resize(img_1, [100, 100])

    # 预测
    data = img.reshape([img.shape[0] * img.shape[1], 3])
    result = np.array(smo.predict(data))
    result = result.reshape([100, 100])
    result = (result + 1) * 125
    result = np.uint8(result)
    result = cv2.resize(result, [size[1], size[0]])        # 得到分辨率与原图一致的图像
    thre, result = cv2.threshold(result, 0, 255, cv2.THRESH_OTSU)
    # 预测结束
    # 膨胀处理
    k = np.ones((5, 5), np.uint8)
    image = cv2.dilate(result, k)
    return image

if __name__ == "__main__":
    # sys.exit(main())

    # 向量机预测前景背景结果显示
    t = time.time()
    a = lambda  x:x
    print(a(t))
    print(time.time()-t)

    # 训练并保存支持向量机的程序
    # main_1()

    # 测试函数img_value
    # a = img_2value(r'D:\sweetvegetable\test\image\bonirob_2016-05-18-10-59-09_4_frame148.png')
    # print(a.shape)