import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import numpy as np
import torchattacks
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import math
from attack import Attack
import imageio
img_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
    transforms.ToTensor()
])
def show_images_diffrence(img, nima_img):
    plt.title('Original_img')
    plt.imshow(img)
    plt.show()

    plt.title('fgsm_img')
    plt.imshow(nima_img)
    plt.show()
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


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFAULT: 0.007)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        attack = torchattacks.FGSM(model, eps=0.007)
        adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=0.007):
        super(FGSM, self).__init__("FGSM", model)
        self.eps = eps

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        score = 0.0
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)

        loss = nn.L1Loss()

        images.requires_grad = True
        outputs = self.model(images)
        outputs = outputs.view(10, 1)
        for j, e in enumerate(outputs, 1):
            score += j * e
        cost = self._targeted * loss(score, labels)

        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
base_model = models.vgg16(pretrained=True)
attack_model = NIMA(base_model)
attack_model.load_state_dict(torch.load('/home/luning/PycharmProjects/mac_code/luning_experimental_code/NIMA/epoch-82.pth'))
attack_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target = 0
target = Variable(torch.Tensor([float(target)]).to(device).long())
image_path = "/home/luning/PycharmProjects/mac_code/luning_experimental_code/NIMA/test_images/kodim11.png"
images = Image.open(image_path).convert('RGB')
images = img_transform(images)
images = images.unsqueeze(dim=0)
list = []
theat = 0.001
for i in range(9):
    attack = FGSM(attack_model, eps=theat)
    adv_images = attack(images, target)
    # adv_images_test = adv_images
    # adv_images = adv_images.squeeze(dim=0)
    # adv_images = adv_images.cpu().detach().numpy()
    # adv_images = np.transpose(adv_images, (1, 2, 0))
    # images = images.squeeze(dim=0)
    # images = images.numpy()
    # images = np.transpose(images, (1, 2, 0))
    # show_images_diffrence(images, adv_images)
    outputs_1 = attack_model(adv_images)
    outputs_1 = outputs_1.view(10, 1)
    scores = 0.0
    for j, e in enumerate(outputs_1, 1):
        scores += j * e
    list.append([theat, scores.detach().numpy()])
    theat = theat + 0.001
print('list = ', list)
# import pandas as pd  # 导入模块

# write = pd.ExcelWriter("test.xlsx")   # 新建xlsx文件。
# df1 = pd.DataFrame([1, 2])
# df1.to_excel(write, sheet_name='Sheet1', index=False)  # 写入文件的Sheet1
#
# df2 = pd.DataFrame([4, 5])
# df2.to_excel(write, sheet_name='Sheet2', index=False)  # 写入文件的Sheet2，且不覆盖
# write.save()  # 这里一定要保存

