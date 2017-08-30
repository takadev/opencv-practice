#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

FILE_NAME = 'capture.jpg'

options = Options()
options.binary_location = '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary'
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--start-fullscreen')
options.add_argument('--window-size=1276,777')
driver = webdriver.Chrome(chrome_options=options)
driver.set_page_load_timeout(15)
driver.get("https://www.google.co.jp")
driver.save_screenshot(FILE_NAME)
driver.quit()

gray_img = cv2.imread(FILE_NAME, cv2.IMREAD_GRAYSCALE)
cv2.imshow("result", gray_img)
cv2.imwrite('result.jpg', gray_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
