import numpy as np
import time
import cv2
import os

def photocut(img,x_start,y_start,x_end,y_end):
    z = img.shape
    if img.ndim>2:
        image = np.zeros([x_end-x_start+1,y_end-y_start+1,z[2]])
        for row in range(x_end-x_start+1):
            for point in range(y_end-y_start+1):
                image[row,point,:] = img[row+x_start,point+y_start,:]
        return image

def cal_mid_photo(img):
    [x,y,z] = img.shape
    left = [0,0,x-1,y//2-1]
    right = [0,y//2,x-1,y-1]
    return left,right

def cut_midphoto(image):
    [l,r] = cal_mid_photo(image)
    left_picture = photocut(image,l[0],l[1],l[2],l[3])
    right_picture = photocut(image,r[0],r[1],r[2],r[3])
    return left_picture,right_picture

def save_img(img):
    #save 2 picture in save_path,it is cut with left and right
    save_path = os.getcwd()+'/picture'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    savel_path = save_path+'/left/'
    saver_path = save_path+'/right/'
    if not os.path.exists(savel_path):
        os.mkdir(savel_path)
    if not os.path.exists(saver_path):
        os.mkdir(saver_path)
    [left_img,right_img] = cut_midphoto(image = img)
    image_name = 'IMG'+ str(int(time.time()*10000)) + '.jpg'
    cv2.imwrite(savel_path+image_name,left_img)
    cv2.imwrite(saver_path+image_name,right_img)
    print('successfully capture the picture: ',savel_path,image_name,saver_path,image_name)
