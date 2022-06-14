import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.autograd import Variable
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 使用GPU进行加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 图像加载以及预处理
image_path = "/home/luning/luning_code/luning/code/picture/trian/0.jpg"
orig = cv2.imread(image_path)  # 读取图像
b, g, r = cv2.split(orig)
orig = cv2.merge([r, g, b])
orig = cv2.resize(orig, (224, 224))
print('orig.shape = ', orig.shape)
img = orig.copy().astype(np.float32)  # 将像素值转换成float类型
mean = [0.485, 0.456, 0.406],
std = [0.229, 0.224, 0.225]
img /= 255.0
img = (img - mean) / std
img = img.transpose(2, 0, 1)  # 交换img的序列
img = np.expand_dims(img, axis=0)
img = Variable(torch.from_numpy(img).to(device).float())

# 使用预测模式 主要影响droupout和BN层的行为
model = models.alexnet(pretrained=True).to(device).eval()

label = np.argmax(model(img).data.cpu().numpy())  # 将图像输入找到标签的最大值即预测值
# print("label={}".format(label))

# 图像数据梯度可以获取
img.requires_grad = True

# 设置为不保存梯度值无法修改
for param in model.parameters():
    param.requires_grad = False  # 不修改网络模型的参数
optimizer = torch.optim.Adam([img])
loss_func = torch.nn.CrossEntropyLoss()
loss_func = loss_func.cuda()
epochs = 200
target = 288
target = Variable(torch.Tensor([float(target)]).to(device).long())
#图像的训练过程
for epoch in range(epochs):
    # 梯度清零
    optimizer.zero_grad()
    output = model(img)
    loss = loss_func(output, target)
    label = np.argmax(output.data.cpu().numpy())
    # 如果定向攻击成功
    if label == target:
        break
    loss.backward()  # 反向传播
    optimizer.step()  # 更新梯度
#处理显示的图像
adv = img.data.cpu().numpy()[0]
adv = adv.transpose(1, 2, 0)
adv = (adv * std) + mean
adv = adv * 255.0
adv = np.clip(adv, 0, 255).astype(np.uint8)
# 对比展现原始图片和对抗样本图片
print('adv.shape = ', adv.shape)
def show_images_diffrence(original_img, original_label, adversarial_img, adversarial_label):
    plt.subplot(131)
    plt.title('Original_img')
    plt.imshow(original_img)
    plt.axis('off')

    plt.subplot(132)
    plt.title('Adversarial_img')
    plt.imshow(adversarial_img)
    plt.axis('off')

    plt.subplot(133)
    plt.title('diffrence')
    difference = adversarial_img - original_img  # 原始图像和对抗样本相应像素的差值
    plt.imshow(difference, cmap=plt.cm.gray)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

print('adv.shape1 = ', adv.shape)
show_images_diffrence(orig, 388, adv, target)
