import yaml
import os
import argparse
import logging
from utils import Dataloader
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from model import DayHeap,Loss

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logfile = 'log.txt'
fh=logging.FileHandler(logfile,encoding='utf-8',mode='a')
fh.setLevel(logging.DEBUG)
ch=logging.StreamHandler()
ch.setLevel(logging.WARNING)
# 定义输出格式
formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)



def train(net,device_,config):
    device = torch.device(device_)
    weight_path = config['weight_path']
    # 加载网络权重，判断权重文件是否存在
    if len(weight_path)>0 and os.path.exists(weight_path):
        t = torch.load(weight_path)
        net.load_state_dict(t)
    else:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    net.to(device)
    # 设置梯度下降方式和学习率下降方式
    assert config['optimizer'] == 'SGD' or config['optimizer'] == 'Adam'
    if config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(),config['lr'],momentum=config['momentum'])
    elif config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), config['lr'])
    assert config['schedual'] == 'cosine' or config['schedual'] == 'StepLR'
    if config['schedual'] == 'cosine':
        lf = lambda x: ((1 - np.cos(x * np.pi / config['epoch'])) / 2) * (1e-15 - 1) + 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    elif config['schedual'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,1,gamma=0.95)
    # 数据加载器
    loader = Dataloader('config/config.yaml')
    # 损失计算函数
    criterial = Loss().to(device)
    # 验证精度初始化
    val_ac = 0

    # 开始训练
    st = 0
    for step in range(config['epoch']):
        loss_ = 0
        print('\nstep: ',step,'/',config['epoch']-1)
        # 设置进度条
        bar = tqdm(loader)
        for i,(x,label) in enumerate(bar):
            x = x.to(device)
            det_label = net(x)                      # 网络正向传播
            loss = criterial(det_label,label)      # 计算损失函数
            logger.info(str(step)+' 损失函数为：'+str(float(loss)))
            optimizer.zero_grad()             # 梯度信息归零
            loss.backward()                   # 反向传播求导
            optimizer.step()                  # 更新权重
            bar.set_postfix({'loss': '{0:1.5f}'.format(float(loss))})
            bar.update(1)
            loss_ += float(loss)              # 损失叠加，用于计算该step的平均损失
            st += 1
            # if st == 40:
            #     optimizer = torch.optim.SGD(net.parameters(), opt['lr'], momentum=opt['momentum'])
        scheduler.step()
        print('\nloss_average: ',loss_/(len(loader)))  # 打印平均损失
        # 保存权重文件
        t = net.state_dict()
        torch.save(t,config['weight_path'])
        # 验证集精度
        pass

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='none', help='initial weights path')
    parser.add_argument('--batch_size', type=int, default=-1, help='batchsize')
    parser.add_argument('--path', type=str, default='none', help='dataset path file with yaml')
    parser.add_argument('--lr', type=float, default=-1.0, help='learning rate')
    parser.add_argument('--epoch', type=int, default=-1, help='train epoch')
    parser.add_argument('--momentum', type=float, default=-1.0, help='momentum')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer')
    return parser.parse_known_args()[0] if known else parser.parse_args()

def setconfig():
    opt = parse_opt(True)
    with open('config/config.yaml','r') as f:
        config = yaml.load(f,yaml.FullLoader)
    for key,value in vars(opt).items():
        if isinstance(value,str):
            if value != 'none':
                config[key] = value
        elif isinstance(value,float) or isinstance(value,int):
            if value >= 0:
                config[key] = value
    return config


def main():
    model = DayHeap()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = setconfig()
    logger.info('开始训练：' + config['weight_path'])
    train(model,device,config)
    logger.info('已完成训练：'+config['weight_path'])

if __name__ == '__main__':
    main()