import torch
import torch.nn as nn
import torch.nn.functional as F
import util
class enhance_net(nn.Module):


    def __init__(self):
        super(enhance_net, self).__init__()
        n = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,6.5,7.0,7.5,8.0]
        sigma = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        bias=[0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]
        s=1
        number_f = 32
        self.sigma = nn.Parameter(torch.FloatTensor([sigma]), True)
        self.n = nn.Parameter(torch.FloatTensor([n]), True)
        self.bias = nn.Parameter(torch.FloatTensor(bias), False)
        self.s = nn.Parameter(torch.FloatTensor(s),True)
        self.relu = nn.PReLU()
        self.norm=nn.BatchNorm2d(3,affine=True)
        self.norm32 = nn.BatchNorm2d(32, affine=True)
        self.norm64 = nn.BatchNorm2d(64, affine=True)
        self.norm128 = nn.BatchNorm2d(128, affine=True)

        self.conv_1=nn.Conv2d(16*3,number_f,3,1,1,bias=True)
        self.conv_2=nn.Conv2d(number_f,number_f*2,3,1,1,bias=True)
        self.conv_3=nn.Conv2d(number_f*2,number_f*4,3,1,1,bias=True)
        self.conv_4=nn.Conv2d(number_f*4,number_f*4,3,1,1,bias=True)
        self.conv_5=nn.Conv2d(number_f*8,number_f*2,3,1,1,bias=True)
        self.conv_6=nn.Conv2d(number_f*4,number_f,3,1,1,bias=True)
        self.conv_7=nn.Conv2d(number_f*2,number_f,3,1,1,bias=True)
        self.conv_1_1=nn.Conv2d(number_f,3,1,1,0,bias=True)

        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self,x):
        r1= torch.pow(x, self.n[0][0]) / ((torch.pow(x, self.n[0][0]) + torch.pow(self.sigma[0][0], self.n[0][0]))+self.bias[0])
        r1 = r1.type(torch.FloatTensor).cuda()
        r1 = self.norm(r1)

        r2 = torch.pow(x, self.n[0][1]) / ((torch.pow(x, self.n[0][1]) + torch.pow(self.sigma[0][1], self.n[0][1]))+self.bias[1])
        r2 = r2.type(torch.FloatTensor).cuda()
        r2 = self.norm(r2)

        r3 = torch.pow(x, self.n[0][2]) / ((torch.pow(x, self.n[0][2]) + torch.pow(self.sigma[0][2], self.n[0][2]))+self.bias[2])
        r3 = r3.type(torch.FloatTensor).cuda()
        r3 = self.norm(r3)

        r4 = torch.pow(x, self.n[0][3]) / ((torch.pow(x, self.n[0][3]) + torch.pow(self.sigma[0][3], self.n[0][3]))+self.bias[3])
        r4 = r4.type(torch.FloatTensor).cuda()
        r4 = self.norm(r4)

        r5 = torch.pow(x, self.n[0][4]) / ((torch.pow(x, self.n[0][4]) + torch.pow(self.sigma[0][4], self.n[0][4]))+self.bias[4])
        r5 = r5.type(torch.FloatTensor).cuda()
        r5 = self.norm(r5)

        r6 = torch.pow(x, self.n[0][5]) / ((torch.pow(x, self.n[0][5]) + torch.pow(self.sigma[0][5], self.n[0][5]))+self.bias[5])
        r6 = r6.type(torch.FloatTensor).cuda()
        r6 = self.norm(r6)

        r7 = torch.pow(x, self.n[0][6]) / ((torch.pow(x, self.n[0][6]) + torch.pow(self.sigma[0][6], self.n[0][6]))+self.bias[6])
        r7 = r7.type(torch.FloatTensor).cuda()
        r7 = self.norm(r7)

        r8 = torch.pow(x, self.n[0][7]) / ((torch.pow(x, self.n[0][7]) + torch.pow(self.sigma[0][7], self.n[0][7]))+self.bias[7])
        r8 = r8.type(torch.FloatTensor).cuda()
        r8 = self.norm(r8)

        r9 = torch.pow(x, self.n[0][8]) / ((torch.pow(x, self.n[0][8]) + torch.pow(self.sigma[0][8], self.n[0][8]))+self.bias[8])
        r9 = r9.type(torch.FloatTensor).cuda()
        r9 = self.norm(r9)

        r10 = torch.pow(x, self.n[0][9]) / ((torch.pow(x, self.n[0][9]) + torch.pow(self.sigma[0][9], self.n[0][9]))+self.bias[9])
        r10 = r10.type(torch.FloatTensor).cuda()
        r10 = self.norm(r10)

        r11 = torch.pow(x, self.n[0][10]) / ((torch.pow(x, self.n[0][10]) + torch.pow(self.sigma[0][10], self.n[0][10]))+self.bias[10])
        r11 = r11.type(torch.FloatTensor).cuda()
        r11 = self.norm(r11)

        r12 = torch.pow(x, self.n[0][11]) / ((torch.pow(x, self.n[0][11]) + torch.pow(self.sigma[0][11], self.n[0][11]))+self.bias[11])
        r12 = r12.type(torch.FloatTensor).cuda()
        r12 = self.norm(r12)

        r13 = torch.pow(x, self.n[0][12]) / ((torch.pow(x, self.n[0][12]) + torch.pow(self.sigma[0][12], self.n[0][12]))+self.bias[12])
        r13 = r13.type(torch.FloatTensor).cuda()
        r13 = self.norm(r13)

        r14 = torch.pow(x, self.n[0][13]) / ((torch.pow(x, self.n[0][13]) + torch.pow(self.sigma[0][13], self.n[0][13]))+self.bias[13])
        r14 = r14.type(torch.FloatTensor).cuda()
        r14 = self.norm(r14)

        r15 = torch.pow(x, self.n[0][14]) / ((torch.pow(x, self.n[0][14]) + torch.pow(self.sigma[0][14], self.n[0][14]))+self.bias[14])
        r15 = r15.type(torch.FloatTensor).cuda()
        r15 = self.norm(r15)

        r16 = torch.pow(x, self.n[0][15]) / ((torch.pow(x, self.n[0][15]) + torch.pow(self.sigma[0][15], self.n[0][15]))+self.bias[15])
        r16 = r16.type(torch.FloatTensor).cuda()
        r16 = self.norm(r16)

        x_fusion=torch.cat([r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16],dim=1)

        x1 = self.norm32(self.maxpool(self.relu(self.conv_1(x_fusion))))
        x2 = self.norm64(self.maxpool(self.relu(self.conv_2(x1))))
        x3 = self.norm128(self.maxpool(self.relu(self.conv_3(x2))))
        x4 = self.norm128(self.conv_4(x3))
        x5 = self.norm64(self.upsample(self.relu(self.conv_5(torch.cat([x4, x3], 1)))))
        x6 = self.norm32(self.upsample(self.relu(self.conv_6(torch.cat([x2, x5], 1)))))
        x_out =torch.sigmoid(self.upsample(self.relu(self.conv_7(torch.cat([x1, x6], 1)))))
        x_out =self.conv_1_1(x_out)

        x_mean=torch.mean(x,dim=1,keepdim=True)
        x_out1=torch.mean(x_out,dim=1,keepdim=True)
        x_result =x_out1 * (x / (x_mean+1e-6))
        return x_result
