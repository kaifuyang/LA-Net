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
        self.name=os.listdir(lowlight_image_path)
        print("Total training examples:", len(self.name))
    def __trans__(self,img):
        img =  cv2.resize(img,(self.size, self.size), interpolation=cv2.INTER_AREA)
        img = (np.asarray(img) / 255.0)
        img=torch.from_numpy(img).float()
        img=img.permute(2,0,1)
        return img
    def __getitem__(self, index):
        self.name.sort(key=lambda x: int(x[:-4]))
        name = self.name[int(index)]
        img = cv2.imread(os.path.join(self.lowlight_image, name))
        gt = cv2.imread(os.path.join(self.ground_truth, name))
        return self.__trans__(img),self.__trans__(gt)
    def __len__(self):
        return len(self.name)
