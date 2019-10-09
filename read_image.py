from PIL import Image
import numpy as np
from PIL import Image
from pylab import *
import os
import glob
 
 
# 训练时所用输入长、宽和通道大小
w = 28
h = 28
c = 3
# 将标签转换成one-hot矢量
def to_one_hot(label):
    label_one = np.zeros((len(label),4))
    for i in range(len(label)):
        label_one[i, label[i]-1]=1
    return label_one
# 读入图片并转化成相应的维度
def read_img(path):
    cate = os.listdir(path)
    imgs   = []
    labels = []
    for folder in cate:
        idx = int(folder)
        fp = os.path.join(path, folder)
        for im in os.listdir(fp):
            im = os.path.join(fp, im)
            # 读入图片，转化成灰度图，并缩小到相应维度
            img = array(Image.open(im).resize((w,h)),dtype=float32)
            # print(img.shape)
            imgs.append(img)
            #img = array(img)
            labels.append(idx)
    print('trainset num ：',len(labels))
    data,label = np.asarray(imgs, np.float32), to_one_hot(np.asarray(labels, np.int32))
    
    # 将图片随机打乱
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]
    # 80%用于训练，20%用于验证
    ratio = 0.8
    s = np.int(num_example * ratio)
    x_train = data[:s]
    y_train = label[:s]
    x_val   = data[s:]
    y_val   = label[s:]
 
    x_train = np.reshape(x_train, [-1, w,h,3])
    x_val = np.reshape(x_val, [-1, w,h,3])
 
    return x_train, y_train, x_val, y_val
 
if __name__=="__main__":
    path = r'I:\课外相关\1\images'
    x_train, y_train, x_val, y_val = read_img(path)
