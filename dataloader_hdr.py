import torch
import numpy as np
import os
import torch.utils.data as data
import cv2
import random


class lowlight_loader(data.Dataset):

    def __init__(self,lowlight_image_path,ground_truth_path):
        self.size=512
        self.lowlight_image=lowlight_image_path
        self.ground_truth=ground_truth_path
        self.name_low=os.listdir(lowlight_image_path)
        self.name_gt = os.listdir(ground_truth_path)

        print("Total training examples:", len(self.name_low*2))

    def __trans__(self,img,gt_flag,gama):
        if gt_flag==1:
            img = cv2.resize(img,(self.size, self.size), interpolation=cv2.INTER_AREA)
            img = (np.asarray(img) / 255.0)
            img = torch.from_numpy(img).float()
            img = img.permute(2,0,1)
        else:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
            img = (np.asarray(img))
            Max = np.max(img)
            img = np.log(img + 1.0)
            A = np.log(Max + 1.0)
            img = np.power(img / A, gama)
            img = torch.from_numpy(img).float()
            img = img.permute(2, 0, 1)
        return img
    def __getitem__(self, index):

        self.name_low.sort(key=lambda x: int(x[:-4]))
        self.name_gt.sort(key=lambda x: int(x[:-4]))

        name_low = self.name_low[int(index/2)]
        name_gt  = self.name_gt[int(index/2)]
        img = cv2.imread(os.path.join(self.lowlight_image, name_low), flags = cv2.IMREAD_ANYDEPTH)
        gt  = cv2.imread(os.path.join(self.ground_truth, name_gt))
        a = random.uniform(0.7,2.0)
        C = 1
        gama = round(a, C)
        return self.__trans__(img,0,gama),self.__trans__(gt,1,gama)

    def __len__(self):
        return len(self.name_low*2)
