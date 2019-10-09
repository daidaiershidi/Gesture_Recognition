from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
import time
import cv2
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
from torchvision import datasets, models, transforms
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
    return dst
def get_one_image(train):
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]
    # img_dir = train[n-1]
    img = cv2.imread(img_dir)
    img = cv2.resize(img, (227, 227))
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    # cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
    # 可选参数alpha是伸缩系数，beta是加到结果上的一个值，结果返回uint类型的图像
    Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
    Scale_absY = cv2.convertScaleAbs(y)
    result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
    # image = Image.open(img_dir)
    # image = image.resize([227, 227])
    image = np.array(result)
    return image

def cam(model):
    import cv2 as cv

    img_roi_y = 30
    img_roi_x = 200
    img_roi_height = 300  # [2]设置ROI区域的高度
    img_roi_width = 300  # [3]设置ROI区域的宽度
    capture = cv.VideoCapture(0)
    index = 1
    num = 0
    while True:
        ret, frame = capture.read()
        if ret is True:
            img_roi = frame[img_roi_y:(img_roi_y + img_roi_height), img_roi_x:(img_roi_x + img_roi_width)]
            # img_roi = ellipse_detect(img_roi)
            # cv.imshow("frame", img_roi)
            # img_roi = cv2.resize(img_roi, (80, 80))
            img_roi = img_roi[50:250,50:250]
            cv2.imshow("frame", img_roi)
            img_roi = img_roi[11:191,11:191]
            index += 1
            if index % 5 == 0:   # 每5帧保存一次图像
                num += 1
                cv.imwrite("pred/pred/" + "camtest."+str(num) + ".jpg", img_roi)
            if num == 1:
                evaluate_one_image(model,np.array(img_roi))
                num = 0
            c = cv.waitKey(50)  # 每50ms判断一下键盘的触发。  0则为无限等待。
            if c == 27:  # 在ASCII码中27表示ESC键，ord函数可以将字符转换为ASCII码。
                break
            if index == 1000:
                break
        else:
            break
    
    cv.destroyAllWindows()
    capture.release()
def evaluate_one_image(model,array):
    num_class = 3
    start = time.time()
    transform2 = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    torch_data = transform2(array)
    torch_data = torch_data.view(-1,3,180,180)
    # data_transforms = {
    #     'pred': transforms.Compose([
    #         transforms.Resize(80),
    #         transforms.CenterCrop(70),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    # }
    # image_datasets = {}
    # dataloders = {}
    # image_datasets['pred']=datasets.ImageFolder('pred', data_transforms['pred'])
    # dataloders['pred']=torch.utils.data.DataLoader(image_datasets['pred'],
    #                                                 batch_size=1,
    #                                                 shuffle=False,
    #                                                 num_workers=1) 
    # data_gen = enumerate(dataloders['pred'])
    output=[]
    # for i, data in data_gen:
    #     input_var = torch.autograd.Variable(data[0])
    #     print(data[0].size())
    #     # compute output
    #     rst= model(input_var)
    #     rst=rst.detach().cpu().numpy().copy()
    #     # measure accuracy and record loss
    #     output.append(rst.reshape(1,num_class))
    input_var = torch.autograd.Variable(torch_data)
    # print(data[0].size())
    # compute output
    rst= model(input_var)
    rst=rst.detach().cpu().numpy().copy()
    # measure accuracy and record loss
    output.append(rst.reshape(1,num_class))
    image_pred = [np.argmax(x[0]) for x in output]
    end = time.time()
    print(len(image_pred))
    print('results is ',idx_to_class[image_pred[0]],' using time ',end-start)


if __name__ == '__main__':
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    model = models.resnet18()
    num_ftrs1 = model.fc.in_features#512
    model.fc = nn.Linear(num_ftrs1, 3)
    model.load_state_dict(torch.load('./checkpoint/ckpt.t7',map_location='cpu'))
    class_to_idx = {'3': 2, '2': 1, '1': 0}
    idx_to_class = {0: '1', 1: '2', 2: '3'}
    cam(model)    
    # evaluate_one_image()




