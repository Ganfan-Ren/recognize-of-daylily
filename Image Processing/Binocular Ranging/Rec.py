import cv2
import numpy as np

def correct_img(Left,Right,left_matrix,right_matrix,distortion_l,distortion_r,R,T):
    RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(left_matrix, distortion_l,
                                                      right_matrix, distortion_r,
                                                      Right.shape[::-1], R, T)
    Left_Map = cv2.initUndistortRectifyMap(left_matrix, distortion_l, RL, PL,
                                           Left.shape[::-1], cv2.CV_16SC2)
    Right_Map = cv2.initUndistortRectifyMap(right_matrix, distortion_r, RR, PR,
                                            Right.shape[::-1], cv2.CV_16SC2)
    Left_nice = cv2.remap(Left, Left_Map[0], Left_Map[1], cv2.INTER_LANCZOS4,
                          cv2.BORDER_CONSTANT, 0)
    Right_nice = cv2.remap(Right, Right_Map[0], Right_Map[1], cv2.INTER_LANCZOS4,
                           cv2.BORDER_CONSTANT, 0)
    return Left_nice,Right_nice

if __name__ == '__main__':
    Left = cv2.imread('../doubleeye/image/left/left01.jpg',0)
    Right = cv2.imread('../doubleeye/image/right/right01.jpg',0)
    left_matrix = np.array([[533.02761,0,342.56959],
                            [0,533.14345,233.80236],
                            [0,0,1]])#左目内参
    right_matrix = np.array([[537.40944,0,324.10550],
                             [0,537.22892,246.87953],
                             [0,0,1]])#右目内参
    distortion_l = np.array([-0.28951,0.09892,0.00120,-0.00012,0])#左畸变
    distortion_r = np.array([-0.29886,0.14455,-0.00048,0.00112,0])#右畸变
    R = np.array([[0.9999,0.0037,0.0102],
         [-0.0037,1.0,-0.0039],
         [-0.0102,0.0039,0.9999]])#旋转矩阵
    T = np.array([-99.69439,1.39820,0.76235])#平移向量
    Left_nice,Right_nice = correct_img(Left,Right,left_matrix,right_matrix,distortion_l,distortion_r,R,T)
    cv2.imshow('left',Left_nice)
    cv2.imshow('Right',Right_nice)
    cv2.imwrite('../doubleeye/image/left01_nice.jpg',Left_nice)
    cv2.imwrite('../doubleeye/image/right01_nice.jpg',Right_nice)
    cv2.waitKey()