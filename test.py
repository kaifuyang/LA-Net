import torch
import numpy as np
import Enhance
import Decom
import Denoise
import util
import os
import time
import glob
import cv2
import argparse

h = 0
w = 0
def lowlight(image_path,config):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda
    if config.train_model=="hdr":
        data_lowlight = cv2.imread(image_path, flags=cv2.IMREAD_ANYDEPTH)
        data_lowlight = np.asarray(data_lowlight)
        Max = np.max(data_lowlight)
        data_lowlight = np.log(data_lowlight + 1.0)
        A = np.log(Max + 1.0)
        data_lowlight = data_lowlight / A
        data_lowlight_shape = data_lowlight.shape
        if data_lowlight_shape[0] % 8 != 0:
            h = data_lowlight_shape[0] - data_lowlight_shape[0] % 8
        else:
            h = data_lowlight_shape[0]
        if data_lowlight_shape[1] % 8 != 0:
            w = data_lowlight_shape[1] - data_lowlight_shape[1] % 8
        else:
            w = data_lowlight_shape[1]
        data_lowlight = data_lowlight[:h, :w, :]
        data_lowlight = torch.from_numpy(data_lowlight).float()
        data_lowlight = data_lowlight.permute(2, 0, 1)
        data_lowlight = torch.unsqueeze(data_lowlight, 0).cuda()
    else:
        data_lowlight = cv2.imread(image_path)
        data_lowlight_shape = data_lowlight.shape
        if data_lowlight_shape[0] % 8 != 0:
            h = data_lowlight_shape[0] - data_lowlight_shape[0] % 8
        else:
            h = data_lowlight_shape[0]
        if data_lowlight_shape[1] % 8 != 0:
            w = data_lowlight_shape[1] - data_lowlight_shape[1] % 8
        else:
            w = data_lowlight_shape[1]
        data_lowlight = data_lowlight[:h, :w, :]
        data_lowlight = (np.asarray(data_lowlight) / 255.0)
        data_lowlight = torch.from_numpy(data_lowlight).float()
        data_lowlight=data_lowlight.permute(2,0,1)
        data_lowlight = torch.unsqueeze(data_lowlight, 0).cuda()


    #model_loading
    La_net=Enhance.enhance_net().cuda()
    decom_net=Decom.decompose_net().cuda()
    DES_net=Denoise.Denois_net().cuda()

    checkpoint = torch.load(config.snapshots_pth)
    La_net.load_state_dict(checkpoint['TYB_net'])
    DES_net.load_state_dict(checkpoint['DES_net'])
    decom_net.load_state_dict(checkpoint['decom_net'])

    img_lowlight_high_frequency, img_lowlight_low_frequency = decom_net(data_lowlight)

    enhance_img_lowlight_low_frequency = La_net(img_lowlight_low_frequency)

    # denoise
    denoised_img_lowlight_high_frequency = DES_net(img_lowlight_high_frequency)
    enhanced_image=enhance_img_lowlight_low_frequency+denoised_img_lowlight_high_frequency

    result_path=image_path.replace('low','result')

    if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
        os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))
    util.save_img(enhanced_image,result_path)

if __name__ == '__main__':
    with torch.no_grad():
        parser = argparse.ArgumentParser()
        parser.add_argument('--test_path', type=str, default="./datasets/LOL/eval15/low")
        parser.add_argument('--train_model', type=str, default="lol")
        parser.add_argument('--cuda', type=str, default="0")
        parser.add_argument('--snapshots_pth', type=str, default="./snapshots/lol/Epoch199.pth")

        config = parser.parse_args()

        file_list = os.listdir(config.test_path)

        for file_name in file_list:

            test_list = glob.glob(config.test_path + file_name )

            for image in test_list:

                lowlight(image,config)




