# Problem 1

import cv2
import sys
import time
import numpy as np

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)

frame_counter = 0
a = time.time()
while True:

    ret, frame = capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = cv2.waitKey(1)
    frame_counter += 1

    if (frame_counter == 1 or frame_counter % 300 == 0):
        corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
        corners = np.int0(corners)
        for i in corners:
            x,y = i.ravel()
            cv2.circle(img,(x,y),3,255,-1)

    cv2.imshow('FaceDetection', img)

    if k % 256 == 27:
        break
    
capture.release()
cv2.destroyAllWindows()
