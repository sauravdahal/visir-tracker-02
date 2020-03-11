# Problem 2 and 3

import cv2
import sys
import time
import numpy as np

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)

feature_params = dict( maxCorners = 100,
            	       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0,255,(100,3))

frame_counter = 0
corners = 0
a = time.time()
ret, old_frame = capture.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(old_gray,25,0.01,10)
mask = np.zeros_like(old_frame)
while True:
    k = cv2.waitKey(1)
    ret, frame = capture.read()
    frame_counter += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (frame_counter % 300 == 0):
        corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
        corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),3,255,-1)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)

    good_new = p1[st==1]
    good_old = p0[st==1]

    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)

    img = cv2.add(frame,mask)
    cv2.imshow('frame',img)
    
    if k % 256 == 27:
        break

    old_gray = gray.copy()
    p0 = good_new.reshape(-1,1,2)

    
capture.release()
cv2.destroyAllWindows()
