# 修改文件
# import torch
# import numpy as np
# import torch
# from torch.autograd import Variable
# from torchvision import models
# from torchvision import transforms
# import numpy as np
# import matplotlib.pyplot as plt
# import torch.nn as nn
# from PIL import Image
# import torch.nn.functional as F
# import vgg
# from skimage import io
# from skimage import img_as_ubyte
# img_transform = transforms.Compose([
#     transforms.ToTensor()
# ])
# def save_image(diffrence_image):
#     io.imsave('/home/luning/PycharmProjects/mac_code/luning_experimental_code/experiment_picture/FGSM/target_1/diffrence_2022.png', img_as_ubyte(diffrence_image))
# image_path_1 = '/home/luning/PycharmProjects/mac_code/luning_experimental_code/experiment_picture/FGSM/target_1/kidom24_new.png'
# image_path_2 = '/home/luning/PycharmProjects/mac_code/luning_experimental_code/experiment_picture/FGSM/target_1/nima_kidom_24_new.png'
# img_1 = Image.open(image_path_1).convert('RGB')
# img_1 = img_transform(img_1)
# img_1 = img_1.numpy()
# img_1 = np.transpose(img_1, (1, 2, 0))
# img_2 = Image.open(image_path_2).convert('RGB')
# img_2 = img_transform(img_2)
# img_2 = img_2.numpy()
# img_2 = np.transpose(img_2, (1, 2, 0))
# difference = (img_2-img_1)*10
# difference = np.clip(difference, -1, 1).astype(np.uint8)
# save_image(difference)
class dataset():
    def __init__(self):
        return

    def abb(self):
        self.print()
    def print(self):
        print('110')
a = dataset()
a.abb()