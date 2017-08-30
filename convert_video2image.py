# -*- coding: utf-8 -*-

import numpy as np
import os
import cv2

cap = cv2.VideoCapture('./test.mp4')

currentFrame = 0
while(True):
    if currentFrame > 2040:
        break

    ret, frame = cap.read()

    name = './frame/' + str(currentFrame) + '.jpg'
    print ('Creating...' + name)
    cv2.imwrite(name, frame)

    currentFrame += 1

cap.release()
cv2.destroyAllWindows()
