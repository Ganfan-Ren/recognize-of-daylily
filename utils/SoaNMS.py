import cv2
import torch
import numpy as np
import torch.nn.functional as F

class Detres(): # detect result
    def __init__(self,res_al,config,threshold):
        self.device = res_al[0].device
        self.res_al = res_al[0]
        self.heapmap = res_al[1]
        self.config = config
        self.threshold = threshold
        self.index = []
        self.isNMS = False

    def NMS(self,SOAthreshold=0.5): # 针对所有类别
        res_thre = [torch.zeros([0,11]).to(self.device),torch.zeros([0,11]).to(self.device),torch.zeros([0,11]).to(self.device)]
        for i,obj in enumerate(self.res_al):
            if obj[0] >= self.threshold:
                res_thre[int(obj[-2])] = torch.cat([res_thre[int(obj[-2])],obj.view(1,11)],0)
                self.index.append(i)
        # 排序+Nms
        x = 0
        for i,c_resthre in enumerate(res_thre):
            res_thre[i] = self.Nms(c_resthre,SOAthreshold)
            x += len(c_resthre)
        # print(x)
        res_thre = torch.cat(res_thre,0)
        # print(len(res_thre))
        self.isNMS = True
        self.index = res_thre[:,-1]
        self.res_al = res_thre
        return res_thre

    def Nms(self,res_thre,SOAthreshold): # 只针对一个类别
        temp, index = torch.sort(res_thre, 0)
        for i, ind in enumerate(index[:, 0]):
            temp[i] = res_thre[ind]
        res_thre = temp
        i = 0
        while i < len(res_thre)-1:
            j = i+1
            while j < len(res_thre):
                if self.SOA(res_thre[i],res_thre[j],self.config['T']) > SOAthreshold: # 越大越接近
                    res_thre = torch.cat([res_thre[:j-1,:],res_thre[j:,:]])
                    continue
                j += 1
            i += 1
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
        acc_alpha = 0 if 2 * torch.max(torch.abs(x1[3:6]-x2[3:6])) > 1 else 1 - 2 * torch.max(torch.abs(x1[3:6]-x2[3:6]))
        SOA = LP_single * acc_alpha
        # print('LP_single and acc_alpha is {:f} and {:f}'.format(float(LP_single),float(acc_alpha)))
        # print('SOA = {:f}'.format(float(SOA)))
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

    def heap_init(self,radius):
        self.heapmap = F.upsample(self.heapmap,size=self.config['size_img'],mode='nearest')
        heap_d1,index1 = F.max_pool2d_with_indices(self.heapmap,(8,8),(8,8))
        heap_d2,index2 = F.max_pool2d_with_indices(heap_d1,(4,4),(4,4))
        self.heapmap = F.max_unpool2d(heap_d2,index2,(4,4),(4,4))
        self.heapmap = F.max_unpool2d(self.heapmap,index1,(8,8),(8,8))
        self.heapmap = torch.where(self.heapmap>self.threshold,1,0)

    def heap_fix(self,radius):
        self.heap_init(radius)
        cal_x = lambda x: int(x[0] + torch.cos(x[2]) * x[3])
        cal_y = lambda x: int(x[1] - torch.sin(x[2]) * x[3])
        cal_xy = lambda x: (cal_x(x), cal_y(x))  # 跟据角度和长度计算坐标
        keypoints = self.getkeypoint()
        delete_list = []
        num_ = [1,2,3,0]
        flush_index = []
        for i,obj_kp in enumerate(keypoints):
            wait = False
            for j in num_:
                p = obj_kp[j]
                n_point,isflush = self.find_nearestpoint(p,j+1,radius)
                if isflush:
                    keypoints[i][j] = n_point
                    flush_index.append(j)

        new_keypoints = []
        for i,obj in enumerate(keypoints):
            if i in delete_list:
                continue
            new_keypoints.append(obj)
        return keypoints

    def find_nearestpoint(self,point,dim,radius):
        cal_length = lambda x: torch.sqrt((x[1][1] - x[0][1]) ** 2 + (x[1][0] - x[0][0]) ** 2)
        h, w = self.heapmap.shape[2:]
        hmin = point[1]-radius if point[1]-radius >= 0 else 0
        hmax = point[1]+radius+1 if point[1]+radius+1 <= h else h
        wmin = point[0]-radius if point[0]-radius >= 0 else 0
        wmax = point[0] + radius+1 if point[0] + radius + 1 <= w else w
        heapmap_r = self.heapmap[0,dim,hmin:hmax,wmin:wmax]
        point_l = heapmap_r.nonzero()
        point_l[:,0] = point_l[:,0] + hmin
        point_l[:, 1] = point_l[:, 1] + wmin
        length = radius
        flush_sign = False
        new_point = point
        for p in point_l:
            l = cal_length([[p[1],p[0]],point])
            if l<length:
                length = l
                new_point = (int(p[1]),int(p[0]))
                flush_sign = True
        return new_point,flush_sign

    def vis_heapmap(self,radius):
        self.heap_init(radius)
        heap_map = self.heapmap[0,1:,:,:]
        image = heap_map[0,:,:]*50 + heap_map[1,:,:]* 100 + heap_map[2,:,:] * 150 + heap_map[3,:,:]*250
        max_ = float(torch.max(image))
        image = image.cpu().numpy()/max_
        image = cv2.resize(image,[480,320],interpolation=cv2.INTER_NEAREST)
        # cv2.imshow('heapmap',image)
        # cv2.waitKey(0)
        return image

