# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import cv2
import scipy.misc as sm
import matplotlib.pyplot as plt
from torch.autograd import Variable

# 类别
classes = ('aima','Ariana','Billie','caiyilin','chunxia','dengziqi','fanbingbing','guajie',
           'Gulinaza','halsey','jiangshuying','Jingaoyin','jisoo','jtx','kristen','linnalian',
           'linyuner','lisa','liuboxin','liuyifei','malaji','nini','ouyangnana','qbhn','Qiwei',
           'quanzhixian','reba','shiyuan','shuiyuanxizi','solar','songhuiqiao','songzhixiao',
           'tangwei','taylor','tongliya','wangfei','wangou','wangzuxian','xinyuan','xscn',
           'xueli','yangmi','yongye','zhangbozhi','zhangmanyu','zhangxinyu','zhangzifeng',
           'zhangziyi','zhengshuang','zhongtiao')


class vgg16_face(nn.Module):
    def __init__(self,num_classes=50):
        super(vgg16_face,self).__init__()
        inplace = True
        self.conv1_1 = nn.Conv2d(3,64,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.relu1_1 = nn.ReLU(inplace)
        self.conv1_2 = nn.Conv2d(64,64,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.relu1_2 = nn.ReLU(inplace)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
            
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU(inplace)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2_2 = nn.ReLU(inplace)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
            
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU(inplace)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3_3 = nn.ReLU(inplace)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
            
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU(inplace)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4_3 = nn.ReLU(inplace)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
            
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU(inplace)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU(inplace)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5_3 = nn.ReLU(inplace)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False) 
            
        self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.relu6 = nn.ReLU(inplace)
        self.drop6 = nn.Dropout(p=0.5)
        
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU(inplace)
        self.drop7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        
    def forward(self,x):
        out = self.conv1_1(x)
        x_conv1 = out
        out = self.relu1_1(out)
        out = self.conv1_2(out)
        out = self.relu1_2(out)
        out = self.pool1(out)
        x_pool1 = out
        
        out = self.conv2_1(out)
        out = self.relu2_1(out)
        out = self.conv2_2(out)
        out = self.relu2_2(out)
        out = self.pool2(out)
        x_pool2 = out
        
        out = self.conv3_1(out)
        out = self.relu3_1(out)
        out = self.conv3_2(out)
        out = self.relu3_2(out)
        out = self.conv3_3(out)
        out = self.relu3_3(out)
        out = self.pool3(out)
        x_pool3 = out
        
        out = self.conv4_1(out)
        out = self.relu4_1(out)
        out = self.conv4_2(out)
        out = self.relu4_2(out)
        out = self.conv4_3(out)
        out = self.relu4_3(out)
        out = self.pool4(out)
        x_pool4 = out
        
        out = self.conv5_1(out)
        out = self.relu5_1(out)
        out = self.conv5_2(out)
        out = self.relu5_2(out)
        out = self.conv5_3(out)
        out = self.relu5_3(out)
        out = self.pool5(out)
        x_pool5 = out
        
        out = out.view(out.size(0),-1)
        
        out = self.fc6(out)
        out = self.relu6(out)
        out = self.fc7(out)
        out = self.relu7(out)
        out = self.fc8(out)
        
        return out, x_pool1, x_pool2, x_pool3, x_pool4, x_pool5


def makeup_recommendation(img_path):
     
    model=torch.load('model/makeup_recommendation_model.pth')
    img = sm.imread(img_path,mode='RGB')
    img = sm.imresize(img,[224,224])
    input_arr = np.float32(img)
    x = torch.from_numpy(input_arr.transpose((2,0,1))) # c,h,w
    x = x.contiguous()
    x = x.view(1, x.size(0), x.size(1), x.size(2))
    x = Variable(x)
    x = torch.tensor(x, dtype=torch.float32)
    out, x_pool1, x_pool2, x_pool3, x_pool4, x_pool5 = model(x)
    # plt.imshow(x_pool2.data.numpy()[0,45]) # plot
    _, predicted = torch.max(out.data, 1)
    return classes[predicted.item()]
