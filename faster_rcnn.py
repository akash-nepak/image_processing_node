import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv 
import torchvision
import math
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_iou(boxes1, boxes2):
#param boxes1: (N,4)
#param boxes2: (M,4)
#return iou: (N,M)
    area1 = (boxes1[:,2]- boxes1[:,0]) * (boxes1[:,3]- boxes1[:,1])
    area2 = (boxes2[:,2]- boxes2[:,0]) * (boxes2[:,3]- boxes2[:,1]) 

    #get top left and bottom right coordinates
    x_left = torch.max(boxes1[:,None,0], boxes2[:,0])
    y_top = torch.max(boxes1[:,None,1], boxes2[:,1])
    x_right = torch.min(boxes1[:,None,2], boxes2[:,2])
    y_bottom = torch.min(boxes1[:,None,3], boxes2[:,3])

    intersection_area = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)
    union = area1[:,None] + area2 - intersection_area
    return intersection_area * 1.0/ union





     

    

def apply_regression_pred_to_anchors_or_proposals(box_transform_pred,anchors_or_proposals):
    #shape of the box_transform_pred = (numofanchors*no_of_class*4)

    box_transform_pred = box_transform_pred.reshape(box_transform_pred.size(0),-1,4)

    #get cx cy w h from x1 y1 x2 y2
    w = anchors_or_proposals[:,2] - anchors_or_proposals[:,0]
    h = anchors_or_proposals[:,3] - anchors_or_proposals[:,1]

    center_x = anchors_or_proposals[:,0]+ 0.5*w
    center_y =anchors_or_proposals[:,1]+ 0.5*h

    dx = box_transform_pred[...,0]
    dy = box_transform_pred[...,1]
    dw = box_transform_pred[...,2]
    dh = box_transform_pred[...,3]
    #dh (numanchors_or_proposals,num_cla sses)

    pred_center_x = dx * w[:,None] + center_x[:,None]
    pred_center_y = dy * h[:,None] + center_y[:,None]
    pred_w = torch.exp(dw) * w[:,None]
    pred_h = torch.exp(dh) * h[:,None]

    pred_box_x1 = pred_center_x - 0.5* pred_w
    pred_box_y1 = pred_center_y - 0.5* pred_h
    pred_box_x2 = pred_center_x + 0.5* pred_w
    pred_box_y2 = pred_center_y + 0.5* pred_h

    pred_box = torch.stack((pred_box_x1,pred_box_y1,pred_box_x2,pred_box_y2),dim =2)

    return pred_box

def clamp_boxes_to_image_size(boxes,image_shape):
    boxes_x1 = boxes[...:,0]
    boxes_y1 = boxes[...:,1]
    boxes_x2 = boxes[...:,2]
    boxes_y2 = boxes[...:,3]

    height,width = image_shape[-2:]
    boxes_x1 = boxes_x1.clamp(min=0,max=width)
    boxes_y1 = boxes_y1.clamp(min=0,max=height)
    boxes_x2 = boxes_x2.clamp(min=0,max=width)
    boxes_y2 = boxes_y2.clamp(min=0,max=height)

    boxes = torch.stack((boxes_x1[...,None],boxes_y1[...,None],boxes_x2[...,None],boxes_y2[...,None]),dim = -1) 

   

    return boxes



    



     
    

class RegionProposalNetwork(nn.Module):

    def __init__(self, in_channels= 512): #takes 512 channel output from cnn backbone
        super().__init__()
        self.scales = [128,256,512]  #scalling of anchor box
        self.aspect_ratios = [0.5,1,2] # 3 aspect ratio for each anchor 
        self.num_anchors = len(self.scales)* len (self.aspect_ratios) #9 anchor box for each cell

        self.rpn_conv = nn.Conv2d(in_channels,in_channels,kernel_size=3,stride =1, padding=1)

        #1x1 conv for classification and prediction only k objecteness instead of 2k
        self.cls_layer = nn.Conv2d(in_channels,self.num_anchors,kernel_size =1,stride=1,)
        #1x1 regression output

        self.bbox_reg_layer = nn.Conv2d(in_channels,self.num_anchors*4,kernel_size = 1, stride =1)
    
    def assign_targets_to_anchors(self,anchors,gt_boxes):

        iou_matrix = get_iou(anchors,gt_boxes) #shape (num_anchors,num_gt_boxes)

        #max iou for each anchor boxes 
        best_match_iou,best_match_gt_index = iou_matrix.max(dim=0)  #shape (num_anchors,)

        best_match_gt__idx_pre_threshold = best_match_gt_index.clone()
        
        




    



        
        
    
    def filter_proposals(self,proposals,cls_scores,image_shape):
        #slecting top k proposals based on objectness score

        cls_score = cls_scores.reshape(-1)
        cls_score = torch.sigmoid(cls_score)
        _, top_n_idx = cls_score.topk(10000)
        cls_score = cls_score[top_n_idx]
        proposals = proposals[top_n_idx]

        #clamp the proposal within the given image
        proposals = clamp_boxes_to_image_size(proposals,image_shape)

        #NMS based on iou of 0.7
        keep_mask = torch.zeros_like(cls_scores,dtype= torch.bool)
        keep_indices = torch.ops.torchvision.nms(proposals,cls_score,0.7)

        post_nms_keep_indices = keep_indices[cls_scores[keep_indices].sort(descending=True)[1]]
       

        #post nms filtering
        proposals = proposals[post_nms_keep_indices[:2000]]
        cls_scores = cls_scores[post_nms_keep_indices[:2000]]
        return proposals,cls_scores
       

       


        



        
    
    
    def generate_anchors(self,image,feat):

        grid_h,grid_w = feat.shape[-2:] #extracts h,w from the feature map
        image_h,image_w =image.shape[-2:]

        stride_h = torch.tensor(image_h // grid_h,dtype= torch.float32,device=feat.device)
        stride_w = torch.tensor(image_w //grid_w,dtype =torch.float32,device= feat.device)
                                
        scales =  torch.as_tensor(self.scales,dtype = feat.dtype,device = feat.device)
        aspect_ratios = torch.as_tensor(self.aspect_ratios,dtype = feat.dtype,device = feat.device)

        #h/w remians 1 i.e aspect ratio remains q all the time 
        h_ratio = torch.sqrt(aspect_ratios )
        w_ratio = 1/h_ratio

        ws = (w_ratio[:,None] * scales[None,:]).view(-1)
        hs = (h_ratio[:,None] * scales[None,:]).view(-1)

        base_anchors = torch.stack([-ws,-hs,ws,hs],dim =1)/2  #centered at 0,0 of the image and the create anchor box for each anchor points
        base_anchors = base_anchors.round()

        #shifts the image in x axis (0,1,....w_feat-1) * stride_h  , stride =16 in original image 
        shifts_x = torch.arange(0,grid_w,dtype=torch.int32,device = feat.device )* stride_w

        #shifts in y axis for anchor boxes
        shifts_y = torch.arange(0,grid_h,dtype=torch.int32,device = feat.device ) *stride_h

        shifts_y,shifts_x = torch.meshgrid(shifts_y,shifts_x,indexing='ij')

        #(H_feat,w_feat)

        shifts_x = shifts_x.reshape(-1)
        shifts_y =shifts_y.reshape(-1)
        shifts = torch.stack((shifts_x,shifts_y,shifts_x,shifts_y),dim =1).to(feat.dtype)

        #shifts -> (H-feat *w_feat,4)
        #base_anchors(h_feat *w-feat ,4)
        anchors = (shifts.view(-1,1,4)+ base_anchors.view(1,-1,4))
        #(h_FEAT * w_feat, num_anchor_per_location,4)

        anchors = anchors.reshape(-1,4)

        return anchors # takes cordinate sfrom top left corners 








         

       
        




    
        
    def forward(self,image,feat,targets):  
        rpn_feat = nn.ReLU()(self.rpn_conv(feat))
        cls_scores = self.cls_layer(rpn_feat)
        box_transform_pred = self.bbox_reg_layer(rpn_feat)

        #generate anchors for the feature maps

        anchors = self.generate_anchors(image,feat)

        #cls_scores (Batch,number of anchors per loactaions,H_feat,W_feat)
        number_of_anchors_per_location = cls_scores.size(1)

        cls_scores = cls_scores.permute(0,2,3,1)
        cls_scores = cls_scores.reshape(-1,1)

        #cls_scores (Batch*h_feat*_w_feat*number_of_anchors_per_loacation,1 )


        

        box_transform_pred = box_transform_pred.view(
            box_transform_pred.size(0),
            number_of_anchors_per_location,
            4,
            rpn_feat.shape[2],
            rpn_feat.shape[-1]
        )
        box_transform_pred = box_transform_pred.permute(0,3,4,1,2)
        box_transform_pred = box_transform_pred.reshape(-1,4)
        #shape (B*H_feat*W_feat*num_ancvhor_locations,4)  

        #transform generated anchors using box_transform_pred
        proposals = apply_regression_pred_to_anchors_or_proposals(box_transform_pred.detach().reshape(-1,1,4),anchors)
        proposals = proposals.reshape(proposals.size(0),4)
        proposals,cls_scores = self.filter_proposals(proposals,cls_scores.detach(),image.shape)
         

        rpn_output = {
            'proposals': proposals,
            'objectness_scores': cls_scores
        }

        if not self.training or targets is None:
            return rpn_output
        else:
            #in training assign gt box and label for each anchor and compute loss



       
        
            
            
            













                                 


