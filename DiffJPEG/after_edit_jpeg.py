import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
# Local
import sys
sys.path.append(r'/home/luning/luning_code/luning/code/DiffJPEG/modules')
import modules_compression,  modules_decompression
from utils import diff_round, quality_to_factor
import cv2
import matplotlib.pyplot as plt
import numpy as np
import smoothing_cnn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DiffJPEG(nn.Module):
    def __init__(self, height=480, width=640, differentiable=True, quality=5):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme.
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = modules_compression.compress_jpeg(rounding=rounding, factor=factor)
        self.decompress = modules_decompression.decompress_jpeg(height, width, rounding=rounding, factor=factor)

    def forward(self, x):
        y, cb, cr = self.compress(x)
        recovered = self.decompress(y, cb, cr)
        return recovered
# 处理图像
class image_transform():
    def __init__(self):
        super(image_transform, self).__init__()

    def image_process(self, image): #处理图像
        diff_img = DiffJPEG()
        image = diff_img(image)
        image = image[0]
        image = image.cpu().detach().numpy()
        image = np.transpose(image, (1, 2, 0))
        return image

    def plt_image(self, image, image_name): #画图
        plt.imshow(image)
        plt.title(image_name)
        plt.axis('off')
        plt.show()

    def conver_rgb(self, image):
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        jpeg_image = cv2.merge([r, g, b])
        return jpeg_image
model = smoothing_cnn.smoothing_network()
model = nn.DataParallel(model) #GPU
model = model.cuda()
model.load_state_dict(torch.load('/home/luning/luning_code/luning/code/pre_edit_cnn.pth'))
weight = 640
height = 480
image_path = '/home/luning/luning_code/luning/code/picture/trian/59.jpg'
img = cv2.imread(image_path)
img = cv2.resize(img, (weight, height))
img = img.copy().astype(np.float32)  # 将像素值转换成float类型
img /= 255.0
img = img.transpose(2, 0, 1)  # 交换img的序列
img = np.expand_dims(img, axis=0)
img = torch.from_numpy(img)
# 压缩 1*3*480*640
transforms = image_transform()
cnn_img = model(img).cpu().detach()
after_diff_img = transforms.image_process(img)
after_diff_img = transforms.conver_rgb(after_diff_img)
after_cnn_img = transforms.image_process(cnn_img)
after_cnn_img = transforms.conver_rgb(after_cnn_img)
transforms.plt_image(after_diff_img, 'diff_img')
transforms.plt_image(after_cnn_img, 'cnn_diff_image')