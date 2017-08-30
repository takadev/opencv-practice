# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os

def init_video(img):
	height, width = img.shape[:2]
	fourcc = cv2.VideoWriter_fourcc(*'MP4V')
	out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (width, height))
	return out

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        pad_w, pad_h = int(0.15 * w), int(0.05 * h)
        cv2.rectangle(img, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), thickness)

def PD_default(filename):
    image = cv2.imread(filename)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    found, w = hog.detectMultiScale(image, hitThreshold = 0, winStride = (8,8), padding = (0, 0), scale = 1.05, finalThreshold = 5)

    draw_detections(image, found)
    out.write(image)

sample = cv2.imread('./frame/1.jpg', cv2.IMREAD_UNCHANGED)
out = init_video(sample)

currentFrame = 1
while(True):
    if currentFrame > 100:
        break

    frame = str(currentFrame)
    print('Creating frame:' + frame)
    PD_default('./frame/' + frame + '.jpg')
    currentFrame += 1

out.release()
cv2.destroyAllWindows()
