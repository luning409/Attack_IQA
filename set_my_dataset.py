from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
from glob import glob
from PIL import Image
height = 480
weight = 640
img_path = '/home/luning/luning_code/luning/code/picture'
image_transform = transforms.Compose([
    transforms.Resize((height, weight), interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
Mydataset = ImageFolder(root=img_path, transform=image_transform)
train_loader = torch.utils.data.DataLoader(Mydataset, batch_size=1, shuffle=True)
dataiter = iter(train_loader)
imgs, labels = next(dataiter)
imgs = imgs[0]
imgs = np.transpose(imgs, (1, 2, 0))
plt.imshow(imgs)
plt.show()