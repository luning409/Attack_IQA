from __future__ import print_function  # 即使在python2.X，使用print就得像python3.X那样加括号使用
import os
import cv2
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out_h, out_w = 1, 1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
        # self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        # 其实这里的localization-network也只是一个普通的CNN+全连接层
        # nn.Conv2d前几个参数为in_channel, out_channel, kennel_size, stride=1, padding=0
        # nn.MaxPool2d前几个参数为kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=64, out_channels=6, kernel_size=1, stride=1)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(6 * out_h * out_w, 32),  # in_features, out_features, bias = True
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)  # 先进入localization层
        xs = xs.view(-1, 6 * out_h * out_h)  # 展开为向量
        theta = self.fc_loc(xs)  # 进入全连接层，得到theta向量
        theta = theta.view(-1, 2, 3)  # 对theta向量进行resize操作，输出2*3的仿射变换矩阵,通道数为C
        # affine_grid函数的输入中，theta的格式为(N,2,3)，size参数的格式为(N,C,W',H')
        # affine_grid函数中得到的输出grid的大小为(N,H,W,2)，这里的2是因为一个点的坐标需要x和y两个数来描述
        grid = F.affine_grid(theta=theta, size=x.size())  # 这里size参数为输出图像的大小，和输入一样，因此采取x.size
        # grid_sample函数的输入中，x代表ST的输入图，格式为(N,C,W,H),W'可以不等于W,H‘可以不等于H;grid是上一步得到的
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        # Perform the usual forward pass
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        # return F.log_softmax(x, dim=1)
        return x

if __name__ == "__main__":
    model = Net().to(device)
    # model.load_state_dict(torch.load('stn_model.pth'))
    # img_path = '/home/luning/luning_code/luning/code/cut_img/0.jpg'
    # img = cv2.imread(img_path)
    # img_b, img_g, img_r = cv2.split(img)
    # img_r = cv2.resize(img_r, (224, 224))
    # img_r = torch.from_numpy(img_r)
    # img_b = cv2.resize(img_b, (224, 224))
    # img_b = torch.from_numpy(img_b)
    # img_g = cv2.resize(img_g, (224, 224))
    # img_g = torch.from_numpy(img_g)
    # out_red = model(img_r)
    # out_green = model(img_g)
    # out_blue = model(img_b)
    # img = cv2.merge([r, g, b])
    # img = cv2.resize(img, (224, 224))
    # img = np.transpose(2, 0, 1)
    # img = np.expand_dims(img, axis=0)
    # output = model(img)