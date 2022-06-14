import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from skimage import io
from skimage import img_as_ubyte
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
import math
import imageio
#NIMA
"""
file - model.py
Implements the aesthemic model and emd loss used in paper.

Copyright (C) Yunxiao Shi 2017 - 2020
NIMA is released under the MIT license. See LICENSE for the fill license text.
"""
img_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
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


def show_images_diffrence(img, nima_img):
    plt.axis('off')
    plt.imshow(img)
    plt.show()

    plt.axis('off')
    plt.imshow(nima_img)
    plt.show()
def single_emd_loss(p, q, r=2):
    """
    Earth Mover's Distance of one sample
    Args:
        p: true distribution of shape num_classes × 1
        q: estimated distribution of shape num_classes × 1
        r: norm parameter
    """
    assert p.shape == q.shape, "Length of the two distribution must be the same"
    length = p.shape[0]
    emd_loss = 0.0
    for i in range(1, length + 1):
        emd_loss += torch.abs(sum(p[:i] - q[:i])) ** r
    return (emd_loss / length) ** (1. / r)


def emd_loss(p, q, r=2):
    """
    Earth Mover's Distance on a batch
    Args:
        p: true distribution of shape mini_batch_size × num_classes × 1
        q: estimated distribution of shape mini_batch_size × num_classes × 1
        r: norm parameters
    """
    assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
    mini_batch_size = p.shape[0]
    loss_vector = []
    for i in range(mini_batch_size):
        loss_vector.append(single_emd_loss(p[i], q[i], r=r))
    return sum(loss_vector) / mini_batch_size


def psnr(target, ref):
    # target:目标图像  ref:参考图像  scale:尺寸大小
    diff = ref - target
    diff = diff.flatten('C')
    mse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(1.0 / mse)
# 保存原始图片和产生的对抗样本的图片
def save_image(orig_image, nima_image):
    io.imsave('/home/luning/PycharmProjects/mac_code/luning_experimental_code/experiment_picture/FGSM/target_1/kidom24_new.png', img_as_ubyte(orig_image))
    io.imsave('/home/luning/PycharmProjects/mac_code/luning_experimental_code/experiment_picture/FGSM/target_1/nima_kidom_24_new.png', img_as_ubyte(nima_image))
    io.imsave('/home/luning/PycharmProjects/mac_code/luning_experimental_code/experiment_picture/FGSM/target_1/difference_24_fgsm.png', img_as_ubyte(orig_image - nima_image))

#模型部分
height = 480
width = 640
score, std = 0.0, 0.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = models.vgg16(pretrained=True)
attack_model = NIMA(base_model)
attack_model.load_state_dict(torch.load('/home/luning/PycharmProjects/mac_code/luning_experimental_code/NIMA/epoch-82.pth'))
attack_model = attack_model.to(device).eval()
target = 1
target = Variable(torch.Tensor([float(target)]).to(device))
if __name__ == '__main__':
    image_path = "/home/luning/PycharmProjects/mac_code/luning_experimental_code/NIMA/test_images/kodim24.png"
    #得到正确的NIMA的得分
    img = Image.open(image_path).convert('RGB')
    img = img_transform(img)
    orig_img = img
    img = img.unsqueeze(dim=0)
    img = Variable(img.to(device))
    img.requires_grad = True
    # 图像数据梯度可以获取
    loss_fun = torch.nn.MSELoss().cuda()
    # 设置为不保存梯度值无法修改
    for param in attack_model.parameters():
        param.requires_grad = False  # 不修改网络模型的参数
    optimizer = torch.optim.Adam([img])
    epochs = 500
    for epoch in range(epochs):
        score = 0.0
        optimizer.zero_grad()
        out = attack_model(img)
        out = out.view(10, 1)
        for j, e in enumerate(out, 1):
            score += j * e
        print('score = ', score)
        loss = loss_fun(target, score)
        loss.backward()
        optimizer.step()
        if loss < 0.001:
            break
        # print('第' + str(epoch) +'迭代\n')
    nima_img = img.cpu().detach().numpy()[0]
    nima_img = np.transpose(nima_img, (1, 2, 0))
    nima_img = np.clip(nima_img, a_min=0.0, a_max=1.0)
    orig_img = torch.transpose(orig_img, 0, 2)
    orig_img = torch.transpose(orig_img, 0, 1)
    orig_img = orig_img.cpu().numpy()
    print('orig_img.shape = ', orig_img.shape)
    # 对比展现原始图片和对抗样本图片
    show_images_diffrence(orig_img, nima_img)
    # 保存原始图片和产生的对抗样本的图片
    save_image(orig_img, nima_img)
    # 计算两幅图像的PSNR的值
    img_psnr = psnr(orig_img, nima_img)
    print('psnr = ', img_psnr)
