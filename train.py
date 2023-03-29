import torch.utils
import Denoise
import Decom
import Enhance
import argparse
import dataloader_LOL
import dataloader_EE
import dataloader_hdr
import os
from tqdm import tqdm
import util

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(config):

    torch.autograd.set_detect_anomaly(True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda
    La_net = Enhance.enhance_net().cuda()
    La_net.apply(weights_init)
    decom_net=Decom.decompose_net().cuda()
    decom_net.apply(weights_init)
    DES_net = Denoise.Denois_net().cuda()
    DES_net.apply(weights_init)
    if config.train_model=="lol":
        train_dataset = dataloader_LOL.lowlight_loader(config.lowlight_images_path, config.ground_truth_path)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)
    elif config.train_model=="ec":
        train_dataset = dataloader_EE.lowlight_loader(config.lowlight_images_path, config.ground_truth_path)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                                   num_workers=config.num_workers, pin_memory=True)
    elif config.train_model=="hdr":
        train_dataset = dataloader_hdr.lowlight_loader(config.lowlight_images_path, config.ground_truth_path)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                                   num_workers=config.num_workers, pin_memory=True)
    optimizer_La = torch.optim.Adam(La_net.parameters(),lr=config.La_lr, weight_decay=config.weight_decay)
    optimizer_DES = torch.optim.Adam(DES_net.parameters(), lr=config.Denoise_lr, weight_decay=config.weight_decay)
    optimizer_decom = torch.optim.Adam(decom_net.parameters(), lr=config.Decom_lr, weight_decay=config.weight_decay)
    StepLR_decom = torch.optim.lr_scheduler.StepLR(optimizer_decom, step_size=config.Step_size, gamma=config.Step_gama)
    StepLR_DES = torch.optim.lr_scheduler.StepLR(optimizer_DES, step_size=config.Step_size, gamma=config.Step_gama)
    StepLR_La= torch.optim.lr_scheduler.StepLR(optimizer_La, step_size=config.Step_size, gamma=config.Step_gama)


    La_net.train()
    DES_net.train()
    decom_net.train()
    iteration = 0
    Pce_loss = util.VGGPerceptualLoss()
    L_TV = util.L_TV()
    L_TV_low=util.L_TV_low()
    l2_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    for epoch in range(config.num_epochs):
        for img_lowlight, ground_truth in tqdm(train_loader):
            iteration += 1
            img_lowlight = img_lowlight.cuda()
            ground_truth = ground_truth.cuda()
            optimizer_La.zero_grad()
            optimizer_DES.zero_grad()
            optimizer_decom.zero_grad()
            #decom_net
            img_lowlight_high_frequency,img_lowlight_low_frequency=decom_net(img_lowlight)
            img_highlight_high_frequency,img_highlight_low_frequency=decom_net(ground_truth)

            img_lowlight_high_frequency_detach=img_lowlight_high_frequency.detach()
            img_highlight_high_frequency_detach=img_highlight_high_frequency.detach()
            img_highlight_low_frequency_detach=img_highlight_low_frequency.detach()

            enhance_img_lowlight_low_frequency= La_net(img_lowlight_low_frequency)

            denoised_img_lowlight_high_frequency = DES_net(img_lowlight_high_frequency_detach)

            #decom_loss
            RE_loss = l2_loss(img_highlight_high_frequency + img_highlight_low_frequency, ground_truth) \
                      + l2_loss(img_lowlight_high_frequency + img_lowlight_low_frequency, img_lowlight)
            TV_loss_decom=L_TV_low(img_lowlight_low_frequency)+L_TV(img_highlight_low_frequency)
            L2_loss_decom = l2_loss(img_lowlight_low_frequency, img_lowlight) + l2_loss(img_highlight_low_frequency,
                                                                                  ground_truth)
            loss_decom =100*RE_loss+2*L2_loss_decom+TV_loss_decom

            #enhance_loss
            L2_loss_enh = l2_loss(enhance_img_lowlight_low_frequency,img_highlight_low_frequency_detach)
            loss_enh=L2_loss_enh

            #denoise_loss
            l1_loss_deno = l1_loss(denoised_img_lowlight_high_frequency, img_highlight_high_frequency_detach)
            loss_deno=l1_loss_deno

            result=denoised_img_lowlight_high_frequency+enhance_img_lowlight_low_frequency

            L2_loss=l2_loss(result,ground_truth)
            pce_loss = Pce_loss(result, ground_truth)
            loss = loss_decom +10* loss_enh + loss_deno+5*L2_loss+pce_loss

            psnr_train = util.batch_PSNR(result, ground_truth, 1.)
            ssim_train = util.SSIM(result, ground_truth)

            loss.backward()
            optimizer_La.step()
            optimizer_DES.step()
            optimizer_decom.step()

            if ((iteration + 1) % config.display_iter) == 0:
                print("epoch:", epoch, ",", "Loss at iteration", iteration + 1, ":", loss.item(), "PSNR:", psnr_train,
                      "SSIM:", ssim_train)

        if ((epoch) % config.snapshot_iter) == 0:
            state = {'TYB_net': La_net.state_dict(),
                     'DES_net': DES_net.state_dict(),
                     'decom_net': decom_net.state_dict()}
            torch.save(state, config.snapshots_folder + "Epoch" + str(epoch) + '.pth')
        StepLR_decom.step()
        if epoch > 50:
            StepLR_DES.step()
            StepLR_La.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lowlight_images_path', type=str,
                        default="/home/cc/CODE/CZC_net_code/data/data/low")
    parser.add_argument('--ground_truth_path', type=str, default="/home/cc/CODE/CZC_net_code/data/data/high")
    parser.add_argument('--train_model', type=str, default="lol")
    parser.add_argument('--cuda', type=str, default="2")
    parser.add_argument('--La_lr', type=float, default=1e-4)
    parser.add_argument('--Decom_lr', type=float, default=2e-4)
    parser.add_argument('--Denoise_lr', type=float, default=1e-4)
    parser.add_argument('--Step_size', type=float, default=50)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--Step_gama', type=float, default=0.5)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--display_iter', type=int, default=1)
    parser.add_argument('--snapshot_iter', type=int, default=1)
    parser.add_argument('--snapshots_folder', type=str, default="/home/cc/CODE/CZC_net_code/snapshots/lol")
    parser.add_argument('--load_pretrain', type=bool, default=False)

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)