import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
from torchvision import models
# 开始引用文件
import smoothing_cnn  # cnn模型
import stn_model  # stn模型
import stn_pre_processing
import vgg
import DiffJPEG.DiffJPEG


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
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out


img_transform = transforms.Compose([
    transforms.ToTensor()
])


def show_images_diffrence(img, nima_img):
    img = torch.squeeze(img, dim=0)
    nima_img = torch.squeeze(nima_img, dim=0)
    img = img.cpu().detach().numpy()
    nima_img = nima_img.cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    nima_img = np.transpose(nima_img, (1, 2, 0))
    plt.axis('off')
    plt.title('orig_img')
    plt.imshow(img)
    plt.show()

    plt.axis('off')
    plt.title('nima_img')
    plt.imshow(nima_img)
    plt.show()

height = 224
weight = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = models.vgg16(pretrained=True)
attack_model = NIMA(base_model).cuda()
attack_model.load_state_dict(torch.load('/home/luning/PycharmProjects/mac_code/luning_experimental_code/NIMA/epoch-82.pth'))
# smoothing_cnn
model_cnn = smoothing_cnn.smoothing_network()
model_cnn = nn.DataParallel(model_cnn)
model_cnn = model_cnn.cuda()
model_cnn.load_state_dict(torch.load('/home/luning/PycharmProjects/mac_code/luning_experimental_code/gauss_loader_model.pth'))
# STN
# model_stn = stn_model.Net()
# model_stn = nn.DataParallel(model_stn)
# model_stn = model_stn.cuda()
# 参数
epochs = 300
leaning_rate = 0.001
# 损失函数
loss_func = torch.nn.MSELoss()
loss_func = loss_func.cuda()
# 优化器
optimizer_cnn = torch.optim.Adam(model_cnn.parameters(), lr=leaning_rate)
# optimizer_stn = torch.optim.Adam(model_stn.parameters(), lr=leaning_rate)
image_path = '/home/luning/PycharmProjects/mac_code/luning_experimental_code/experiment_picture/FGSM/target_1/nima_bird.png'
score = 0.0
img = Image.open(image_path).convert('RGB')
img = img_transform(img)
img = img.unsqueeze(dim=0)
img = Variable(img.to(device))
img.requires_grad = True
out_image = attack_model(img)
out = out_image.view(10, 1)
for j, e in enumerate(out, 1):
    score += j * e
origin_score = score  # 图片的原始NIMA分数
print('origin_score = ', origin_score)
model_cnn.train()
# 训练网络
for epoch in range(epochs):
    score = 0.0
    optimizer_cnn.zero_grad()
    # optimizer_stn.zero_grad()
    # 经过CNN处理后的图像为img1，要求出入是tensor，1*3*480*640
    img1 = model_cnn(img)
    img1 = img1.cuda()
    # 放入JPEG进行压缩 要求1*3*480*640 而且要把tensor拿到cpu才能进入JPEG压缩
    diff_fun = DiffJPEG.DiffJPEG.DiffJPEG().to('cuda')
    img1 = diff_fun(img1)
    out_score = attack_model(img1)
    out_score = out_score.view(10, 1)
    for j, e in enumerate(out_score, 1):
        score = score + (j * e)
    print('origin_score = ', origin_score)
    print('score = ', score)
    #vgg的输入输出都是tensor 3*480*640
    # img_vgg = vgg.FeatureVisualization(img, 2).show_feature_to_img()
    # img1_vgg = vgg.FeatureVisualization(img1, 2).show_feature_to_img()
    loss = loss_func(origin_score, score) + loss_func(img, img1)
    show_images_diffrence(img, img1)
    print('loss = ', loss)
    loss.backward(retain_graph=True)
    optimizer_cnn.step()
    print('第', epoch+1, '次迭代\n')
torch.save(model_cnn.state_dict(), 'correct_nima_score.pth')
