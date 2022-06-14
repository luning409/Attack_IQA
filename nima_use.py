from PIL import Image
import torchvision.models as models
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from nima_model import *
from torch.autograd import Variable
base_model = models.vgg16(pretrained=True)
model = NIMA(base_model)
model.load_state_dict(torch.load('/home/luning/PycharmProjects/mac_code/luning_experimental_code/NIMA/epoch-82.pth'))
seed = 42
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
#
#
model.eval()
def test_transform(img):
    img = np.array(img)
    # img = img.transpose(1, 2, 0)
    img = cv2.resize(img, (224, 224))
    # plt.imshow(img)
    # plt.show()
    img = img.transpose(2, 0, 1)
    return img

def nima_score(img):
    mean, std, sum_e = 0.0, 0.0, 0.0
    img = test_transform(img)
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img).float().cuda()
    with torch.no_grad():
        out = model(img)
    out = out.view(10, 1)
    print('out = ', out)
    for j, e in enumerate(out, 1):
        sum_e += e
        mean += j * e
    print('sum_e = ', sum_e)
    return mean
if __name__ == '__main__':
    img_path = '/home/luning/PycharmProjects/mac_code/dataset/bird/n01530575_47.JPEG'
    img = Image.open(img_path).convert('RGB')
    img_score = nima_score(img)
    print('nima_score = ', img_score)