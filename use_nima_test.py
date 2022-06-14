from PIL import Image
import torchvision.models as models
import numpy as np
import torch
import matplotlib.pyplot as plt
from nima_model import *
import torchvision.transforms as transforms
base_model = models.vgg16(pretrained=True)
model = NIMA(base_model)
model.load_state_dict(torch.load('/home/luning/PycharmProjects/mac_code/luning_experimental_code/NIMA/epoch-82.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
test_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
    transforms.ToTensor()
    ])

score_list = []
for i in range(100):
    mean = 0.0
    image = Image.open('/home/luning/PycharmProjects/mac_code/luning_experimental_code/dataset/' + str(i+1) + '.png').convert('RGB')
    image = test_transform(image)
    image = image.unsqueeze(dim=0)
    image = image.to(device)
    with torch.no_grad():
        out = model(image)
    out = out.view(10, 1)
    for j, e in enumerate(out, 1):
        mean += j * e
    score_list.append(round(mean.cpu().detach().numpy().item(), 3))
print('score_list = ', score_list)
score_sum = 0.0
for i in range(len(score_list)):
    score_sum += score_list[i]
average = round(score_sum/100, 3)
plt.rc('font', family='Times New Roman')
print('average = ', average)
plt.hist(score_list, edgecolor = 'black')
plt.title("Data Score Distribution")
plt.xlabel("score", family='Times New Roman')
plt.ylabel("number")
ax = plt.gca()
ax.set_ylim(0,30)

plt.show()