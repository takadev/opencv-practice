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
driver.get("https://www.google.co.jp/search?safe=off&biw=1021&bih=544&tbm=isch&sa=1&q=%E3%82%A2%E3%82%B9%E3%83%AA%E3%83%BC%E3%83%88+%E4%B8%80%E8%A6%A7&oq=%E3%82%A2%E3%82%B9%E3%83%AA%E3%83%BC%E3%83%88+%E4%B8%80%E8%A6%A7&gs_l=psy-ab.3..0i24k1.29655.31042.0.31428.8.8.0.0.0.0.108.643.5j2.7.0....0...1.1j4.64.psy-ab..2.4.383...0j0i4k1j0i4i24k1j0i5i30k1j0i8i30k1.m0ilZvl7Yn8")
driver.save_screenshot(FILE_NAME)
driver.quit()

face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

org = cv2.imread(FILE_NAME, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray)

# オリジナルのサイズを保存しておく。
#  shapeで取得できるサイズとresizeの引数に渡すサイズでは横縦の順番が違うらしい。ので[::-1]として反転。
size = org.shape[:2][::-1]

# 一旦1/10にリサイズします
resize = cv2.resize(org, (int(size[0]/10), int(size[1]/10)))

# リサイズした画像をオリジナルサイズに戻します。
# これを「モザイク」として使います。
mozaik = cv2.resize(resize, size, interpolation = cv2.INTER_NEAREST)

for x,y,w,h in faces:
	mozaikFace = mozaik[y:y+h, x:x+w]
	org[y:y+h, x:x+w] = mozaikFace
	cv2.rectangle(org, (x,y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('mozaik', org)
cv2.imwrite('mozaik.jpg', org)

cv2.waitKey(0)
cv2.destroyAllWindows()
