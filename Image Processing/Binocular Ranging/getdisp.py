#!usr/bin/env/ python
# _*_ coding:utf-8 _*_

import cv2 as cv
import numpy as np
import os

#选取左右文件夹里的第一张照片来测试
def get_pic(train_dir2, train_dir3):
    pic_file2 = os.listdir(train_dir2)
    pic_data2 = cv.imread(os.path.join(train_dir2, pic_file2[0]))
    pic_file3 = os.listdir(train_dir3)
    pic_data3 = cv.imread(os.path.join(train_dir3, pic_file3[0]))

    return pic_data2, pic_data3

def get_f_r(pos, arr):#找到arr向量往左大于0的第一个数的坐标l，往右数大于0坐标r
    l = r = pos
    for i in range(pos, 1, -1):
        if arr[i] > 0:
            l = i
            break
    for i in range(pos, len(arr)-1):
        if arr[i] > 0:
            r = i
            break

    return l, r

def getdisp(imgL,imgR):
    min_disp = 0
    num_disp = 112 - min_disp
    window_size = 9
    stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=7,
                                   P1=8 * 1 * window_size ** 2,
                                   P2=32 * 1 * window_size ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32
                                   )

    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    return disp
if __name__ == '__main__':
    left_dir2 = r'F:\PycharmProject\pythonProject\doubleeye\image\left'
    right_dir3 = r'F:\PycharmProject\pythonProject\doubleeye\image\right'

    dst, dst2 = get_pic(left_dir2, right_dir3)
    #转换为灰度图后计算视差图
    imgL = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    imgR = cv.cvtColor(dst2, cv.COLOR_BGR2GRAY)
    disp = getdisp(imgL,imgR)
    #至此视差图就得到了，下面为处理黑边和显示视差图的过程

    min_disp = 0
    num_disp = 112 - min_disp
    new_disp = (disp - min_disp) / num_disp

    count = 0
    for j in range(new_disp.shape[0]):
        if new_disp[0, j] > 0:
            count = j
            break
    new_disp_pp = np.zeros_like(new_disp)
    for i in range(new_disp.shape[0]):
        for j in range(count, new_disp.shape[1] - 1):
            if new_disp[i, j] <= 0:
                left, right = get_f_r(j, new_disp[i])
                #up, down  = get_f_r(i, new_disp[:, j])
                new_disp_pp[i, j] = (new_disp[i, left - 1] + new_disp[i, right + 1]) / 2
            else:
                new_disp_pp[i, j] = new_disp[i, j]
    cv.medianBlur(new_disp_pp,5, new_disp_pp)

    cv.imshow('org_img', dst)
    cv.imshow('org_disp', new_disp)
    cv.imshow("disp_pp", new_disp_pp)
    cv.waitKey()
    '''
    right_stereo = cv.StereoSGBM_create(minDisparity=-num_disp,
                                  numDisparities=num_disp,
                                  blockSize=8,
                                  P1=8 * 3 * window_size ** 2,
                                  P2=32 * 3 * window_size ** 2,
                                  disp12MaxDiff=1,
                                  uniquenessRatio=10,
                                  speckleWindowSize=100,
                                  speckleRange=32
                                  )

    right_disp1 = right_stereo.compute(imgR, imgL).astype(np.float32) / 16.0
    right_new_disp = -(right_disp1 - min_disp) / num_disp

    for i in range(375):
        for j in range(1242):
            if right_new_disp[i, j] >= 1:
                right_new_disp[i, j] = 0

    right_new_disp = cv.medianBlur(right_new_disp, 5)
    '''

    cv.waitKey()
    cv.destroyAllWindows()
