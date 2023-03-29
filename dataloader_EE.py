import torch
import numpy as np
import os
import torch.utils.data as data
import cv2
class lowlight_loader(data.Dataset):
    def __init__(self,lowlight_image_path,ground_truth_path):
        self.size=512
        self.lowlight_image=lowlight_image_path
        self.ground_truth=ground_truth_path
        self.name_img=os.listdir(lowlight_image_path)
        self.name_gt = os.listdir(ground_truth_path)
        print("Total training examples:", len(self.name_img))

    def __trans__(self,img,flag):
       # print(2)
        img =  cv2.resize(img,(self.size, self.size), interpolation=cv2.INTER_AREA)
        img = (np.asarray(img) / 255.0)
        img = torch.from_numpy(img).float()
        img=img.permute(2,0,1)
        return img
    def __getitem__(self, index):
        self.name_img.sort(key=lambda x: int(x[1:5]))
        self.name_gt.sort(key=lambda x: int(x[1:5]))
        name = self.name_img[index]
        name_gt=self.name_gt[int(index/5)]
        img = cv2.imread(os.path.join(self.lowlight_image, name))
        gt = cv2.imread(os.path.join(self.ground_truth, name_gt))
        return self.__trans__(img,1),self.__trans__(gt,0)
    def __len__(self):
        return len(self.name_img)
