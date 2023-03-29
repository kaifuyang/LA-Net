import numpy as np
from skimage.measure.simple_metrics import compare_psnr
import torch
from skimage.metrics import structural_similarity
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import cv2

class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1
                 ):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2)
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2)
        return self.TVLoss_weight *  (h_tv.sum() / count_h + w_tv.sum() / count_w) / batch_size
class L_TV_low(nn.Module):
    def __init__(self, TVLoss_weight=5):
        super(L_TV_low, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow(((x[:, :, 1:, :] - x[:, :, :h_x - 1, :])), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight *  (h_tv / count_h + w_tv / count_w) / batch_size

def SSIM(x_image, y_image, max_value=1.0, win_size=3, use_sample_covariance=True):
    x_image = x_image.permute(0, 2, 3, 1)
    y_image = y_image.permute(0, 2, 3, 1)
    x = x_image.data.cpu().numpy().astype(np.float32)
    y = y_image.data.cpu().numpy().astype(np.float32)
    ssim=0
    for i in range(x.shape[0]):
        ssim += structural_similarity(x[i,:,:,:],y[i,:,:,:], win_size=win_size, data_range=max_value, multichannel=True)
    return (ssim/x.shape[0])

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks).cuda()
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)).cuda()
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)).cuda()
        self.resize = resize

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss



def save_img(img,img_path):
    enhanced_image = torch.squeeze(img, 0)
    enhanced_image = enhanced_image.permute(1, 2, 0)
    enhanced_image = np.asarray(enhanced_image.cpu())
    enhanced_image = enhanced_image * 255.0
    cv2.imwrite(img_path, enhanced_image)








