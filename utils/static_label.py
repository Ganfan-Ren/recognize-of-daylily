import os
from utils.fileop import File
import numpy as np
import matplotlib.pyplot as plt

def staticlong(path,size):
    pl = os.listdir(path)
    label_ = File()
    cal_length = lambda x: np.sqrt((x[1][1] - x[0][1]) ** 2 + (x[1][0] - x[0][0]) ** 2)
    long_ = []
    for pa in pl:
        path_l = '/'.join([path,pa])
        label_.read(path_l)
        keypoint = label_.getpointfloat()
        for k,obj in keypoint.items():
            obj = np.array(obj)
            obj[:,0] = obj[:,0] * size[1]
            obj[:,1] = obj[:,1] * size[0]
            long = cal_length([obj[1],obj[2]])
            long_.append(long)
    return np.array(long_)

def kmeans(src,center,iter_num,stopcondition):
    # 只提供一维的
    if len(src.shape) == 1:
        dim = 1
    elif len(src.shape) == 2:
        dim = src.shape[1]
    else:
        raise UserWarning("数据结构不正确")
    callength = lambda x: np.abs(x[0]-x[1])  # 改变这里可对高维度进行聚类
    c = len(center)

    for i in range(iter_num):
        long_c = []
        for j in range(c):
            long_c.append([])  # 初始化
        for long in src:
            length_index = np.argmin(np.array([callength([long,center[k]]) for k in range(c)]))
            long_c[length_index].append(long)
        new_center = []
        sum = 0
        for j in range(c):
            long_c[j] = np.array(long_c[j])
            av = np.mean(long_c[j])
            new_center.append(av)
            sum += callength([av,center[j]])
        if sum < stopcondition:
            return new_center,long_c
        else:
            center = new_center
    return center,long_c

def vis_static(long):
    min = np.min(long)
    max = np.max(long)
    start = int(min/0.5) * 0.5
    end = (int(max/0.5)+1) * 0.5
    num = []
    n = int((end-start)/0.5)
    static_num = lambda x: np.sum(np.where(np.logical_and(x[0]>=x[1], x[0] < x[2]),1,0))
    y = []
    x = []
    for i in range(n):
        num.append(str(start+i*0.5)+'~'+str(start+i*0.5+0.5))
        y.append(static_num([long,start+i*0.5,start+i*0.5+0.5]))
        x.append(i)
    plt.bar(x,y,align="center",color='b',tick_label=num,alpha=0.6)
    plt.xlabel("范围")
    plt.ylabel("目标数量")
    plt.grid(True,axis="y",alpha=0.1,color="r")
    plt.show()

if __name__ == '__main__':
    long_allobj = staticlong(r'D:\dataset\label',[320,480])
    center,long = kmeans(long_allobj,[5,40],10,1e-5)
    print(center)
    vis_static(long[1]/center[1])

    print(np.min(long[0]/center[0]),np.max(long[0]/center[0]))
    print(np.min(long[1] / center[1]), np.max(long[1] / center[1]))