import numpy as np

class File():
    def __init__(self):
        self.path = ''
        self.dict = {}
        self.oblist = []
    def read(self,path):
        self.path = path
        with open(self.path,'r') as f:
            info = f.read()
        picinfo = info.split('\n\n')

        self.name = picinfo[0].split(' ')[0]
        self.imageh = int(picinfo[0].split(' ')[1])
        self.imagew = int(picinfo[0].split(' ')[2])

        intinfo = picinfo[1].split('\n')
        self.intinfo_list = []
        for int_ in intinfo:
            int_info = int_.split(' ')[:-1]
            a = []
            for num in int_info:
                a.append(int(num))
            self.intinfo_list.append(a)
        self.intinfo_list = np.array(self.intinfo_list)

        floatinfo = picinfo[2].split('\n')
        self.floatinfo_list = []
        for float_ in floatinfo:
            if len(float_) == 0:
                break
            float_info = float_.split(' ')[:-1]
            a = []
            for num in float_info:
                a.append(float(num))
            self.floatinfo_list.append(a)
        self.floatinfo_list = np.array(self.floatinfo_list)

        index_str  = picinfo[3].split(' ')[:-1]
        index_int = index_str
        for i,ind in enumerate(index_str):
            index_int[i] = int(ind)

        self.oblist = self.intinfo_list[:,:4]
        self.dict = self.floatinfo_list[:,4:]
        self.index = index_int


    def getboxint(self):
        a = {}
        for i,info in enumerate(self.oblist):
            a[self.index[i]] = info.reshape(-1,2)
        return a

    def getpointint(self):
        a = {}
        for i, info in enumerate(self.intinfo_list[:,4:]):
            a[self.index[i]] = info.reshape(-1, 2)
        return a

    def getpointfloat(self):
        a = {}
        for i, info in enumerate(self.dict):
            obj = info.reshape(-1, 2)
            obj[:,0] = obj[:,0] * self.imageh / self.imagew
            obj[:,1] = obj[:,1] * self.imagew / self.imageh
            a[self.index[i]] = obj
        return a

    def getboxfloat(self):
        a = {}
        for i, info in enumerate(self.floatinfo_list[:,:4]):
            obj = info.reshape(-1, 2)
            obj[:, 0] = obj[:, 0] * self.imageh / self.imagew
            obj[:, 1] = obj[:, 1] * self.imagew / self.imageh
            a[self.index[i]] = obj
        return a

    def getdict(self):
        dic = {}
        point = self.getpointfloat()
        box = self.getboxfloat()
        for i,ind in enumerate(self.index):
            po = point[ind]  # 4个点
            bo = box[ind]
            pp = []  # [[x,y],[x,y],...,[x,y]]
            for j,p in enumerate(po):  # 1个点[x,y]
                x_min,y_min,x_max,y_max = bo[0,0],bo[0,1],bo[1,0],bo[1,1]
                x = (p[0] - x_min) / (x_max - x_min)
                y = (p[1] - y_min) / (y_max - y_min)
                pp.append([x,y])
            dic[ind] = pp
        return dic

    def get_imgsize(self):
        try:
            return [self.h,self.w]
        except:
            raise UserWarning("请先读取文件")