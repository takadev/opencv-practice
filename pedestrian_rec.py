# -*- coding: utf-8 -*-

import numpy as np
import cv2
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

DRIVER = './chromedriver'
FILE_NAME = 'capture.jpg'

options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--start-fullscreen')
options.add_argument('--window-size=1276,777')
driver = webdriver.Chrome(DRIVER, chrome_options=options)
driver.set_page_load_timeout(15)
driver.get("https://www.google.co.jp/search?safe=off&biw=1021&bih=544&tbm=isch&sa=1&q=%E6%AD%A9%E8%A1%8C%E8%80%85&oq=%E6%AD%A9%E8%A1%8C%E8%80%85&gs_l=psy-ab.3..0l8.5261.5261.0.5455.1.1.0.0.0.0.87.87.1.1.0....0...1.1.64.psy-ab..0.1.86.vma3h12nqH8")
driver.save_screenshot(FILE_NAME)
driver.quit()

# サンプル顔認識特徴量ファイル
cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/lbpcascades/lbpcascade_animeface.xml')
img = cv2.imread(FILE_NAME)

# HoG特徴量の計算
hog = cv2.HOGDescriptor()
# SVMによる人検出
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 1.05}
# 人を検出した座標
human, r = hog.detectMultiScale(img, **hogParams)

# 長方形で人を囲う
for (x, y, w, h) in human:
	cv2.rectangle(img, (x, y),(x+w, y+h),(0,50,255), 3)

# 画像表示
cv2.imshow('pedestrian_detected',img)
cv2.imwrite('pedestrian_detected.jpg',img)

cv2.waitKey(0)
cv2.destroyAllWindows()