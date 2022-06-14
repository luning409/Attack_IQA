import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from glob import glob
from preprocessing import Dataset


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


def gauss_noise(image_noise, mean=0, var=0.1):  # mean和var分别是均值和方差
    noise = np.random.normal(mean, var ** 0.5, image_noise.shape)
    out = image_noise + noise
    if out.min() < 0:
        low_clip = -1
    else:
        low_clip = 0
    out = np.clip(out, low_clip, 1.0)
    return out
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
height = 480
weight = 640
img_path = '/home/luning/luning_code/luning/code/picture/trian'
image = glob(img_path+'/*'+'.jpg')
image_transform = transforms.Compose([
    transforms.Resize((height, weight), interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
Mydataset = Dataset(image, image_transform)
trian_loader = torch.utils.data.DataLoader(
    Mydataset,
    batch_size=1,
    shuffle=True,
    pin_memory=True
)
model = smoothing_network()
model = nn.DataParallel(model)
model = model.cuda()
# 迭代200次大概跑一天半
epochs = 200
leaning_rate = 0.0001
loss_list = []
#图像梯度可获取
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=leaning_rate)
for epoch in range(epochs):
    model.train()
    for i, img in tqdm(enumerate(trian_loader), total=len(trian_loader)):
        #给图像加入随机的高斯噪声
        random_var = np.random.uniform(0, 0.15)
        img_noise = gauss_noise(img[0], 0, random_var)
        img = img.cuda()
        img.requires_grad = True
        optimizer.zero_grad()
        output = model(img).cuda()
        loss = loss_func(output, img)
        loss_list.append(loss.data)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新梯度
    if epoch > 800:
        if loss[-1] - loss[-2] < 0.001:
            break
    print('第', epoch, '次迭代\n')
torch.save(model.state_dict(), 'gauss_loader_model.pth')




