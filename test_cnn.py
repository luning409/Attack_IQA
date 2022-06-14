import torch
import cv2
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import smoothing_cnn
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight = 480
height = 640
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
model = smoothing_cnn.smoothing_network()
model = nn.DataParallel(model) #GPU
model = model.cuda()
model.load_state_dict(torch.load('pre_edit_cnn.pth'))
image_path = '/home/luning/luning_code/luning/code/picture/trian/1000.jpg'
img = cv2.imread(image_path)
img = cv2.resize(img, (height, weight), interpolation=cv2.INTER_CUBIC)
plt.title('orig_img')
plt.imshow(img)
plt.show()
b, g, r = cv2.split(img)
img = cv2.merge([r, g, b])
img = cv2.resize(img, (height, weight), interpolation=cv2.INTER_CUBIC)
img = img.copy().astype(np.float32)
img /= 255.0
img = (img - mean) / std
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)
img = Variable(torch.from_numpy(img).to(device).float())
img_smooth = model(img)
img_smooth = img_smooth.data.cpu().numpy()[0]
img_smooth = np.transpose(img_smooth, (1, 2, 0))
# print(img_smooth.shape)

for i in range(3):
    __img = img_smooth[:, :, i]
    _mean = mean[i]
    _var = std[i]
    __img = __img * _var + _mean
    img_smooth[:, :, i] = __img
img_smooth *= 255
img_smooth = np.clip(img_smooth, 0, 255).astype(np.uint8)
b, g, r = cv2.split(img_smooth)
img_smooth = cv2.merge([r, g, b])
plt.title('smoothing_img')
plt.imshow(img_smooth)
plt.show()
print('完成')
# print('img_soomthing = ', img_smooth)
