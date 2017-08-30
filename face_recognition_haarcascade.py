# -*- coding: utf-8 -*-
# 画像からカスケード分類器を用いて顔認識を行うサンプル

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
driver.get("https://www.google.co.jp/search?safe=off&biw=1021&bih=544&tbm=isch&sa=1&q=%E3%82%A2%E3%82%B9%E3%83%AA%E3%83%BC%E3%83%88+%E4%B8%80%E8%A6%A7&oq=%E3%82%A2%E3%82%B9%E3%83%AA%E3%83%BC%E3%83%88+%E4%B8%80%E8%A6%A7&gs_l=psy-ab.3..0i24k1.29655.31042.0.31428.8.8.0.0.0.0.108.643.5j2.7.0....0...1.1j4.64.psy-ab..2.4.383...0j0i4k1j0i4i24k1j0i5i30k1j0i8i30k1.m0ilZvl7Yn8")
driver.save_screenshot(FILE_NAME)
driver.quit()

# サンプル顔認識特徴量ファイル
face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_eye.xml')

# 画像の読み込み
img = cv2.imread(FILE_NAME)

# グレースケール変換
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray)

for (x,y,w,h) in faces:
    # 検知した顔を矩形で囲む
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    # 顔画像（グレースケール）
    roi_gray = gray[y:y+h, x:x+w]
    # 顔ｇ増（カラースケール）
    roi_color = img[y:y+h, x:x+w]
    # 顔の中から目を検知
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        # 検知した目を矩形で囲む
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

# 画像表示
cv2.imshow('detected',img)
cv2.imwrite('detected.jpg',img)

cv2.waitKey(0)
cv2.destroyAllWindows()