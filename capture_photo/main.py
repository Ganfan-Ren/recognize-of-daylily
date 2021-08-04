import Photocut
import cv2

print('press \'s\': capture\npress \'q\': exit\n')

cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
while(1):
    ret,picture = cap.read()
    reSize = cv2.resize(picture, (2560//2,720//2), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('capture photo',reSize)
    k = cv2.waitKey(10)
    if k == ord('s'):
        Photocut.save_img(picture)
    elif k == ord('q'):
        break
