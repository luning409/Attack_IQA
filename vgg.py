import torch
from torchvision import models
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch.nn as nn
height = 480
weight = 640
img_mean = torch.tensor([0.485, 0.456, 0.406])
img_std = torch.tensor([0.229, 0.224, 0.225])
image_transform = transforms.Compose([
    transforms.Resize((height, weight), interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=img_mean, std=img_std)
])
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
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
base_model = models.vgg16(pretrained=True)
model = NIMA(base_model).cuda()
model.load_state_dict(torch.load('/home/luning/PycharmProjects/mac_code/luning_experimental_code/NIMA/epoch-82.pth'))
def preprocess_image(img):
    img = Variable(img, requires_grad=True)
    return img
class FeatureVisualization:
    def __init__(self, img, selected_layer):
        self.img = img
        self.selected_layer = selected_layer
        self.pretrained_model = model.features

    def process_image(self):
        img = preprocess_image(self.img)
        return img

    def get_feature(self):
        input = self.process_image()
        x = input
        for index, layer in enumerate(self.pretrained_model):
            x = layer(x)
            if (index == self.selected_layer):
                return x

    def get_single_feature(self):
        features = self.get_feature()
        feature = features[:, 0, :, :]
        feature = feature.view(feature.shape[1], feature.shape[2])
        return feature

    def show_feature_to_img(self):
        feature = self.get_single_feature()
        # use sigmod to [0,1]
        feature = 1.0 / (1 + torch.exp_(-1 * feature))
        feature = torch.round_(feature * 255)
        return feature


if __name__ == '__main__':
    # get class
    img = Image.open('/home/luning/PycharmProjects/mac_code/luning_experimental_code/DiffJPEG/bird.jpg')
    img = image_transform(img)
    img = torch.unsqueeze(img, dim=0)
    img = img.cuda()
    a = FeatureVisualization(img, 2).show_feature_to_img()
    a = a.cpu().detach().numpy()
    plt.imshow(a)
    plt.axis('off')
    plt.show()