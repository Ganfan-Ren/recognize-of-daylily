import Photocut
import cv2

print('press \'s\': capture\npress \'q\': exit\n')

cap=cv2.VideoCapture(0)
while(1):
    ret,picture = cap.read()
    cv2.imshow('picture',picture)
    k = cv2.waitKey(10)
    if k == ord('s'):
        Photocut.save_img(picture)
    elif k == ord('q'):
        break
