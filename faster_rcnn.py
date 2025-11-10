import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv 
import torchvision
import math
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RegionProposalNetwork(nn.Module):

    def __init__(self, in_channels= 512): #takes 512 channel output from cnn backbone
        super().__init__()
        self.scales = [128,256,512]  #scalling of anchor box
        self.aspect_ratio = [0.5,1,2] # 3 aspect ratio for each anchor 
        self.num_anchors = len(self.scales)* len (self.aspect_ratio) #9 anchor box for each cell

        self.rpn_conv = nn.Conv2d(in_channels,in_channels,kernel_size=3,stride =1, padding=1)

        #1x1 conv for classification and prediction only k objecteness instead of 2k
        self.cls_layer = nn.Conv2d(in_channels,self.num_anchors,kernel_size =1,stride=1,)
        #1x1 regression output

        self.bbox_reg_layer = nn.Conv2d(in_channels,self.num_anchors*4,kernel_size = 1, stride =1)

    def generate_anchors(self,image,feat):

        grid_h,grid_w = feat.shape[-2:]
        

        
    def forward(self,image,feat,targets):  
        rpn_feat = nn.ReLU()(self.rpn_conv(feat))
        cls_score = self.cls_layer(rpn_feat)
        box_trasnsform_pred = self.bbox_reg_layer(rpn_feat)

        #generate anchors for the feature maps

        anchors = self.generate_anchors(image,feat)





                                 


