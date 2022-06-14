import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from glob import glob
from torch.autograd import Variable
# 开始引用文件
from preprocessing import Dataset
import smoothing_cnn  # cnn模型
import stn_model  # stn模型
import stn_pre_processing
import vgg
import nima_use
import DiffJPEG.DiffJPEG
height = 480
weight = 640
img_mean = torch.tensor([0.485, 0.456, 0.406])
img_std = torch.tensor([0.229, 0.224, 0.225])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = '/home/luning/luning_code/luning/code/picture/trian'
image = glob(image_path + '/*' + '.jpg')
image_transform = transforms.Compose([
    transforms.Resize((height, weight), interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=img_mean, std=img_std)
])
Mydataset = Dataset(image, image_transform)
train_loader = torch.utils.data.DataLoader(
    Mydataset,
    batch_size=1,
    shuffle=True,
    pin_memory=True
)

# smoothing_cnn
model_cnn = smoothing_cnn.smoothing_network()
model_cnn = nn.DataParallel(model_cnn)
model_cnn = model_cnn.cuda()
model_cnn.load_state_dict(torch.load('gauss_loader_model.pth'))
# STN
# model_stn = stn_model.Net()
# model_stn = nn.DataParallel(model_stn)
# model_stn = model_stn.cuda()
# 参数
epochs = 300
leaning_rate = 0.0001
# 损失函数
loss_func = torch.nn.L1Loss()
loss_func = loss_func.cuda()
# 优化器
optimizer_cnn = torch.optim.Adam(model_cnn.parameters(), lr=leaning_rate)
# optimizer_stn = torch.optim.Adam(model_stn.parameters(), lr=leaning_rate)
# 训练网络
for epoch in range(epochs):
    model_cnn.train()
    # model_stn.train()
    for i, img in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer_cnn.zero_grad()
        # optimizer_stn.zero_grad()
        img.requires_grad = True
        img = img.cuda()
        # 经过CNN处理后的图像为img1，要求出入是tensor，1*3*480*640
        img1 = model_cnn(img)
        img1 = torch.squeeze(img1)
        # STN要求tensor 3*480*640
        # img1 = stn_pre_processing.process_img_stn(img1)
        # img1从stn出来是3*480*640
        #STN不能转化为numpy，他的输入和输出必须都是tensor
        # 放入JPEG进行压缩 要求1*3*480*640 而且要把tensor拿到cpu才能进入JPEG压缩
        img1 = torch.unsqueeze(img1, dim=0)
        img1 = Variable(img1.cpu().float())
        diff_fun = DiffJPEG.DiffJPEG.DiffJPEG()
        img1 = diff_fun(img1)
        img1 = img1.cuda()
        #vgg的输入输出都是tensor 3*480*640
        img_vgg = vgg.FeatureVisualization(img, 2).show_feature_to_img()
        img1_vgg = vgg.FeatureVisualization(img1, 2).show_feature_to_img()
        # NIMA
        img1 = torch.squeeze(img1, dim=0)
        nima_score = nima_use.nima_score(img1)
        loss = loss_func(img_vgg, img1_vgg) - 0.02 * nima_score
        loss.requires_grad_(True)
        loss.backward()
        optimizer_cnn.step()
        # optimizer_stn.step()
    print('第', epoch+1, '次迭代\n')
torch.save(model_cnn.state_dict(), 'pre_edit_cnn.pth')
