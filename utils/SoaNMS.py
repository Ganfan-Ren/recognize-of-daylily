import torch
import numpy as np

class Detres(): # detect result
    def __init__(self,res_al,config,threshold):
        self.res_al = res_al[0]
        self.heaepmap = res_al[1]
        self.config = config
        self.threshold = threshold
        self.index = []
        self.isNMS = False

    def NMS(self,SOAthreshold=0.5):
        res_thre = []
        for i,obj in enumerate(self.res_al):
            if obj[0] >= self.threshold:
                res_thre.append(obj.unsqueeze(0))
                self.index.append(i)
        # 排序
        res_temp = torch.cat(res_thre,0)
        res_thre,index = res_temp.sort(0,True) # True 为从大到小
        ind_temp = []
        for i in index:
            ind_temp.append(self.index[i[0]])
        self.index = ind_temp
        res_tensor = []
        i = 0
        while i < len(res_thre)-1:
            j = i+1
            while j < len(res_thre):
                if self.SOA(res_thre[i],res_thre[j],self.config['T']) > SOAthreshold: # 越大越接近
                    res_thre = torch.cat([res_thre[:j-1,:],res_thre[j:,:]])
                    self.index.pop(j)
                    continue
                j += 1
            i += 1
        self.isNMS = True
        self.res_al = res_thre
        return res_thre


    def SOA(self,x1,x2,T=0.5): # [obj_conf, x, y, angle0, angle1, angle2, length0, length1, length2]
        # info=[x,y,angle,length]             O —— ——>x
        # x = x + torch.cos(angle) * length   |
        # y = y - torch.sin(angle) * length  y|
        cal_x = lambda x: int(x[0] + torch.cos(x[2]) * x[3])
        cal_y = lambda x: int(x[1] - torch.sin(x[2]) * x[3])
        cal_xy = lambda x:(cal_x(x),cal_y(x))  # 跟据角度和长度计算坐标
        cal_length = lambda x:np.sqrt((x[1][1] - x[0][1])**2 + (x[1][0] - x[0][0])**2)
        cal_sigma = lambda x: 0 if x < T else 1
        (x1_2,y1_2) = (int(x1[2]),int(x1[1]))
        (x1_1,y1_1) = cal_xy([x1_2,y1_2,x1[4]-np.pi,x1[7]])
        (x1_3, y1_3) = cal_xy([x1_2, y1_2, x1[3], x1[6]])
        (x1_4, y1_4) = cal_xy([x1_3, y1_3, x1[5], x1[8]])
        keypoint1 = [(x1_1,y1_1),(x1_2,y1_2),(x1_3, y1_3),(x1_4, y1_4)]
        long1 = torch.sum(x1[6:])
        (x2_2, y2_2) = (int(x2[2]), int(x2[1]))
        (x2_1, y2_1) = cal_xy([x2_2, y2_2, x2[4] - np.pi, x2[7]])
        (x2_3, y2_3) = cal_xy([x2_2, y2_2, x2[3], x2[6]])
        (x2_4, y2_4) = cal_xy([x2_3, y2_3, x2[5], x2[8]])
        keypoint2 = [(x2_1, y2_1),(x2_2, y2_2),(x2_3, y2_3),(x2_4, y2_4)]
        long2 = torch.sum(x2[6:])
        long = long1 if x1[0] > x2[0] else long2
        sumd_ij = 0
        for i,p in enumerate(keypoint1):
            sumd_ij += cal_length([p,keypoint2[i]])
        LP_single = torch.exp(-sumd_ij/long*cal_sigma(sumd_ij/long))
        acc_alpha = 0 if 5 * torch.max(torch.abs(x1[3:6]-x2[3:6])) > 1 else 1 - 5 * torch.max(torch.abs(x1[3:6]-x2[3:6]))
        SOA = LP_single * acc_alpha
        return SOA

    def getkeypoint(self):
        if not self.isNMS:
            self.NMS(self.config['SOA'])
        keypoints = []
        cal_x = lambda x: int(x[0] + torch.cos(x[2]) * x[3])
        cal_y = lambda x: int(x[1] - torch.sin(x[2]) * x[3])
        cal_xy = lambda x: (cal_x(x), cal_y(x))  # 跟据角度和长度计算坐标
        for i,obj in enumerate(self.res_al):
            (x2, y2) = (int(obj[2]), int(obj[1]))
            (x1, y1) = cal_xy([x2, y2, obj[4] - np.pi, obj[7]])
            (x3, y3) = cal_xy([x2, y2, obj[3], obj[6]])
            (x4, y4) = cal_xy([x3, y3, obj[5], obj[8]])
            keypoint = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            keypoints.append(keypoint)
        return keypoints