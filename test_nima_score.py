import torch
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
import math
img_transform = transforms.Compose([
    transforms.ToTensor()
])


class NIMA(nn.Module):
    """Neural IMage Assessment model by Google"""
    def __init__(self, base_model, num_classes=10):
        super(NIMA, self).__init__()
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=25088, out_features=num_classes),
            nn.Softmax())

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def psnr(target, ref):
    # target:目标图像  ref:参考图像  scale:尺寸大小
    print('targe.shape = ', target.shape)
    print('ref.shape = ', ref.shape)
    diff = ref - target
    diff = diff.flatten('C')
    mse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(1.0 / mse)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = models.vgg16(pretrained=True)
attack_model = NIMA(base_model).cuda()
attack_model.load_state_dict(torch.load('/home/luning/PycharmProjects/mac_code/luning_experimental_code/NIMA/epoch-82.pth'))
image_path = '/home/luning/PycharmProjects/mac_code/luning_experimental_code/experiment_picture/STN/nima_kodim24.png'
score = 0.0
img = Image.open(image_path).convert('RGB')
img = img_transform(img)
img = img.unsqueeze(dim=0)
img = Variable(img.to(device))
out_image = attack_model(img)
out = out_image.view(10, 1)
for j, e in enumerate(out, 1):
    score += j * e
print('score = ', score)
# 计算两幅图像的PSNR的值
# print('img.shape = ', img.shape)
# print('out_image.shape = ', out_image.shape)
# img_psnr = psnr(img, out_image)
# print('psnr = ', img_psnr)



