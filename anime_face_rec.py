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
driver.get("https://www.google.co.jp/search?q=%E5%A4%A9%E3%80%85%E5%BA%A7%E7%90%86%E4%B8%96&safe=off&source=lnms&tbm=isch&sa=X&ved=0ahUKEwietYe0s_7VAhWIbbwKHZgCDIAQ_AUICigB&biw=1021&bih=544")
driver.save_screenshot(FILE_NAME)
driver.quit()

# サンプル顔認識特徴量ファイル
cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/lbpcascades/lbpcascade_animeface.xml')
img = cv2.imread(FILE_NAME)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
faces = cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (24, 24))

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# 画像表示
cv2.imshow('anime_detected',img)
cv2.imwrite('anime_detected.jpg',img)

cv2.waitKey(0)
cv2.destroyAllWindows()