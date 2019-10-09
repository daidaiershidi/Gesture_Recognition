from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import argparse
import torch.backends.cudnn as cudnn
import cv2
import numpy as np
img_roi_y = 30
img_roi_x = 200
img_roi_height = 300  # [2]设置ROI区域的高度
img_roi_width = 300  # [3]设置ROI区域的宽度
capture = cv2.VideoCapture(0)
index = 1
def _remove_background(frame):
    fgbg = cv2.createBackgroundSubtractorMOG2() # 利用BackgroundSubtractorMOG2算法消除背景
    # fgmask = bgModel.apply(frame)
    fgmask = fgbg.apply(frame)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morpholo
    # gyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((4, 4), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

# 视频数据的人体皮肤检测
def _bodyskin_detetc(frame):
    # 肤色检测: YCrCb之Cr分量 + OTSU二值化
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb) # 分解为YUV图像,得到CR分量
    (_, cr, _) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0) # 高斯滤波
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # OTSU图像二值化
    return skin
def ellipse_detect(image):
    img = image
    skinCrCbHist = np.zeros((256,256), dtype= np.uint8 )
    cv2.ellipse(skinCrCbHist ,(113,155),(23,15),43,0, 360, (255,255,255),-1)
 
    YCRCB = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    (y,cr,cb)= cv2.split(YCRCB)
    skin = np.zeros(cr.shape, dtype=np.uint8)
    (x,y)= cr.shape
    for i in range(0,x):
        for j in range(0,y):
            CR= YCRCB[i,j,1]
            CB= YCRCB[i,j,2]
            if skinCrCbHist [CR,CB]>0:
                skin[i,j]= 255
    dst = cv2.bitwise_and(img,img,mask= skin)

    # Matrix = np.ones((6, 6), np.uint8)
    # dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, Matrix)
    # dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, Matrix)
    return dst
def getedge(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    # cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
    # 可选参数alpha是伸缩系数，beta是加到结果上的一个值，结果返回uint类型的图像
    Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
    Scale_absY = cv2.convertScaleAbs(y)
    result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
    return result
def get_yellow(img):
    lower_yellow = np.array([15, 55, 55])
    upper_yellow = np.array([50, 255, 255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    output = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # 根据阈值找到对应颜色
    # output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    return output
num = 0
while True:
    ret, frame = capture.read()
    if ret is True:
        img_roi = frame[img_roi_y:(img_roi_y + img_roi_height), img_roi_x:(img_roi_x + img_roi_width)]
        # img_roi = _bodyskin_detetc(img_roi)
        # img_roi = ellipse_detect(img_roi)
        # img_roi = _remove_background(img_roi)
        # img_roi = getedge(img_roi)
        # img_roi = cv2.fastNlMeansDenoisingColored(img_roi,None,10,10,7,21)
        # img_roi = get_yellow(img_roi)
        img_roi = img_roi[50:250,50:250]
        cv2.imshow("frame", img_roi)
        img_roi = img_roi[11:191,11:191]
        # b=img_roi[:, :, 0]
        # g=img_roi[:, :, 1]
        # r=img_roi[:, :, 2]
        # cv2.imshow('B',b)
        # cv2.imshow('G', g)
        # cv2.imshow('R', r)
        
        index += 1
        if index % 1 == 0:   # 每5帧保存一次图像
            num += 1
            print(index,end='\r')
            cv2.imwrite("train/3/gesture_3_"+str(num) + ".jpg", img_roi)
        c = cv2.waitKey(50)  # 每50ms判断一下键盘的触发。  0则为无限等待。
        if c == 27:  # 在ASCII码中27表示ESC键，ord函数可以将字符转换为ASCII码。
            break
        if index == 500:
            break
    else:
        break

cv2.destroyAllWindows()
capture.release()
