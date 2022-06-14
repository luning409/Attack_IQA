import torch
import cv2
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.functional as F
import matplotlib.pyplot as plt

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        start_value = x
        out1 = self.residual(x)
        out = start_value + out1
        return out
class smoothing_network(nn.Module):
    def __init__(self):
        super(smoothing_network, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.2)
        )

        self.layer2 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.layer5 = nn.Conv2d(128, 3, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out1 + out2)
        out4 = self.layer4(out3)
        out = self.layer5(out4)
        return out
#用时注释掉，训练时在解开
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# weight = 480
# height = 640
# mean = [0.485, 0.456, 0.406],
# std = [0.229, 0.224, 0.225]
# img_list = []
# for number in range(1000):
#     image_path = '/home/luning/luning_code/luning/code/picture/trian/' + str(number) +'.jpg'
#     img = cv2.imread(image_path)
#     b, g, r = cv2.split(img)
#     img = cv2.merge([r, g, b])
#     img = cv2.resize(img, (height, weight))
#     img = img.copy().astype(np.float32)
#     img /= 255.0
#     img = (img - mean) / std
#     img = img.transpose(2, 0, 1) #交换img的序列
#     img = np.expand_dims(img, axis=0)
#     img = Variable(torch.from_numpy(img).to(device).float())
#     img_list.append(img)
# model = smoothing_network()
# model = nn.DataParallel(model) #GPU
# model = model.cuda()
# model.train()
# epochs = 500
# batch_size = 1
# leaning_rate = 0.0001
# # #图像梯度可获取
# loss_func = torch.nn.MSELoss()
# lost_list = []
# optimizer = torch.optim.Adam(model.parameter(), lr=leaning_rate)
# for epoch in range(epochs):
#     for i in range(len(img_list)):
#         img = img_list[i]
#         # img = torch.from_numpy(img)
#         img.requires_grad = True
#         optimizer.zero_grad()
#         output = model(img)
#         loss = loss_func(output, img)
#         loss.backward()  # 反向传播
#         optimizer.step()  # 更新梯度
#         print('第', epoch, '次', "_", i)
# torch.save(model.state_dict(), 'model.pth')




