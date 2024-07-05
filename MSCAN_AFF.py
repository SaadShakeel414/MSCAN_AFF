# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 23:13:10 2021

@author: Administrator
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


###########################Two-stream Attention Network####################
class Mul_Scale_Channel_Attention(nn.Module):
    def __init__(self,channels=512, reduction = 4, channels_c = 128, reduction_c = 2):
        super(Mul_Scale_Channel_Attention,self).__init__()
       
#        self.convad = nn.Conv2d(channels,128,(1,1),padding=0)
        self.conva1 = nn.Conv2d(channels,channels // reduction,(1,1),padding=0)
        self.conva2 = nn.Conv2d(channels,channels // reduction,(3,3),dilation=3,padding=3)
        self.conva3 = nn.Conv2d(channels,channels // reduction,(3,3),dilation=5,padding=5)
        self.conva4 = nn.Conv2d(channels,channels // reduction,(3,3),dilation=7,padding=7)
        self.relu = nn.ReLU(inplace = True)
        self.softmax = nn.Softmax(dim = -1)
#        self.bna = nn.BatchNorm2d(512)
        self.dense_xx = nn.Conv2d(128,512,kernel_size=1,padding=0,bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = -1)
#        self.bna = nn.BatchNorm2d(256)
        
        self.network = nn.Sequential(nn.Conv2d(channels_c, channels_c // reduction_c , 1, padding = 0, bias = True),
                                     nn.ReLU(inplace = True), nn.Conv2d(channels_c // reduction_c, channels_c, 1, padding=0, bias=True)
                                     ,nn.Softmax(dim = -1))
 
        
    def forward(self, x):

        b,c,w,h = x.size()
    ################First two features####################
        a1 = self.relu(self.conva1(x)).view(b,-1,w,h)        
        a2 = self.relu(self.conva2(x)).view(b,-1,w,h)
        fms12 = a1 + a2
        wei12 = self.sigmoid(fms12)   #64*128*7*7
#        print(wei12)
        rfms12 = wei12 * fms12        ###Refined Multi-scale concatenated feature###

     ###########Last two features##############   
        a3 = self.relu(self.conva3(x)).view(b,-1,w,h)        
        a4 = self.relu(self.conva4(x)).view(b,-1,w,h)
        fms34 = a3 + a4
        wei34 = self.sigmoid(fms34)
        rfms34 = wei34 * fms34       
        
        ############Channel Attention (1,2)############
        rfms12 = self.avg_pool(rfms12)  ###64*128*1*1
        rfms12 = self.network(rfms12)
#        rfms12 = self.bna(rfms12)
        
#        rfms12 = self.softmax(rfms12)
#        rfms12 = self.dense(rfms12)
        f1 = rfms12 * a1      
        f2 = rfms12 * a2
    
        fus_fea12 = f1 + f2
#        fus_fea12 = self.dense_xx(fus_fea12)
        
        
        ############Channel Attention (3,4)############
        rfms34 = self.avg_pool(rfms34)
        rfms34 = self.network(rfms34)
#        rfms34 = self.bna(rfms34)
#        rfms34 = self.softmax(rfms34)
#        rfms34 = self.dense(rfms34)
        f3 = rfms34 * a3     
        f4 = rfms34 * a4
    
        fus_fea34 = f3 + f4
#        fus_fea34 = self.dense_xx(fus_fea34)
        
        fus_fea = fus_fea12 + fus_fea34
        fus_fea = self.dense_xx(fus_fea)
        
        return fus_fea

       
# ---------------------------------- LResNet50E-IR network Begin ----------------------------------

class BlockIR(nn.Module):
    def __init__(self, inplanes, planes, stride, dim_match):
        super(BlockIR, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu1 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        if dim_match:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class LResNet_MFR(nn.Module):

    def __init__(self, block, layers, filter_list, is_gray=False):
        self.inplanes = 64
        super(LResNet_MFR, self).__init__()
        # input is (mini-batch,3 or 1,112,96)
        # use (conv3x3, stride=1, padding=1) instead of (conv7x7, stride=2, padding=3)
        if is_gray:
            self.conv1 = nn.Conv2d(1, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)  # gray
        else:
            self.conv1 = nn.Conv2d(3, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter_list[0])
        self.prelu1 = nn.PReLU(filter_list[0])
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.MFR_Att =  Mul_Scale_Channel_Attention(512)
        self.sigmoid = nn.Sigmoid()
        
        
        self.fc = nn.Sequential(
            nn.BatchNorm1d(512 * 7 * 7),
            nn.Dropout(p=0.4),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512),  # fix gamma ???
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)


    def _make_layer(self, block, inplanes, planes, blocks, stride):
        layers = []
        layers.append(block(inplanes, planes, stride, False))
        for i in range(1, blocks):
            layers.append(block(planes, planes, stride=1, dim_match=True))

        return nn.Sequential(*layers)
    
    

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.MFR_Att(x)
        

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


def LResNet50E_IR_MFR(is_gray=False):
    filter_list = [64, 64, 128, 256, 512]
    layers = [3, 4, 14, 3]
    return LResNet_MFR(BlockIR, layers, filter_list, is_gray)
# ---------------------------------- LResNet50E-IR network End ----------------------------------
