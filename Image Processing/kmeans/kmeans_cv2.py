import cv2
import numpy as np

def trans_type(img):
    m = img.shape[1] * img.shape[0]
    img_line = img.reshape(m, img.shape[2])
    img_line = np.float32(img_line)
    return img_line

def kmeans(path='../kmeans/python-kmeans-dominant-colors-master/images/huanghuacai.jpg',
           img = None,
           criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0),
           isimshow=True,#是否显示聚类图片
           K=5,flags=cv2.KMEANS_PP_CENTERS):
    if img == None:
        img = cv2.imread(path)
    img_line = trans_type(img)
    retval,bestlabels,centers = cv2.kmeans(img_line,K,None,criteria,10,flags)
    kmeans_result = np.zeros([img.shape[0],img.shape[1],img.shape[2]])
    nn = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            kmeans_result[i][j] = centers[bestlabels[nn]]
            nn += 1
    kmeans_result = np.uint8(kmeans_result)
    if isimshow:
        cv2.imshow(path,kmeans_result)
        cv2.waitKey()
    return kmeans_result

if __name__ == '__main__':
    img_path = '../kmeans/python-kmeans-dominant-colors-master/images/huanghuacai'
    file_type = '.jpg'
    K = 4
    iters = 20
    file_name = img_path+'K_'+str(K)+file_type
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iters, 1.0)
    img = kmeans(path=img_path+file_type,K=K,criteria=criteria)
    cv2.imwrite(file_name,img)


