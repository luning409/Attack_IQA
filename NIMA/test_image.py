# import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from model.model import *
base_model = models.vgg16(pretrained=True)
model = NIMA(base_model)
model.load_state_dict(torch.load('epoch-82.pth'))
seed = 42
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
#
#
model.eval()
test_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
    transforms.ToTensor()
    ])
mean, std = 0.0, 0.0
# length = 24
score_list = []
for img in range(9):
    im = Image.open(os.path.join('/home/luning/luning_code/luning/code/picture/trian/' + str(img) + '.jpg'))
    im = im.convert('RGB')
    imt = test_transform(im)
    imt = imt.unsqueeze(dim=0)
    imt = imt.to(device)
    # print(imt.shape)
    with torch.no_grad():
        out = model(imt)
    out = out.view(10, 1)
    # print('out =', out)
    for j, e in enumerate(out, 1):
        mean += j * e
    for k, e in enumerate(out, 1):
        std += e * (k - mean) ** 2
    std = std ** 0.5
    score_list.append(mean)
    mean, std = 0.0, 0.0
for i in range(len(score_list)):
    print('mean =', score_list[i])