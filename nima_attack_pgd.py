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

    plt.title('pgd_img')
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
def single_emd_loss(p, q, r=2):
    """
    Earth Mover's Distance of one sample
    Args:
        p: true distribution of shape num_classes × 1
        q: estimated distribution of shape num_classes × 1
        r: norm parameter
    """
    assert p.shape == q.shape, "Length of the two distribution must be the same"
    length = p.shape[0]
    emd_loss = 0.0
    for i in range(1, length + 1):
        emd_loss += torch.abs(sum(p[:i] - q[:i])) ** r
    return (emd_loss / length) ** (1. / r)


def emd_loss(p, q, r=2):
    """
    Earth Mover's Distance on a batch
    Args:
        p: true distribution of shape mini_batch_size × num_classes × 1
        q: estimated distribution of shape mini_batch_size × num_classes × 1
        r: norm parameters
    """
    assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
    mini_batch_size = p.shape[0]
    loss_vector = []
    for i in range(mini_batch_size):
        loss_vector.append(single_emd_loss(p[i], q[i], r=r))
    return sum(loss_vector) / mini_batch_size


def psnr(target, ref):
    # target:目标图像  ref:参考图像  scale:尺寸大小
    diff = ref - target
    diff = diff.flatten('C')
    mse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(1.0 / mse)
class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFALUT: 0.3)
        alpha (float): step size. (DEFALUT: 2/255)
        steps (int): number of steps. (DEFALUT: 40)
        random_start (bool): using random initialization of delta. (DEFAULT: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=False)
        adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=0.3, alpha=2 / 255, steps=40, random_start=False):
        super(PGD, self).__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)
        loss = nn.L1Loss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        for i in range(self.steps):
            score = 0.0
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            outputs = outputs.view(10, 1)
            for j, e in enumerate(outputs, 1):
                score += j * e
            print('score = ', score)
            print('labels = ', labels)
            cost = loss(score, labels)
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() - self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            if(cost < 0.1):
                break
        return adv_images
base_model = models.vgg16(pretrained=True)
attack_model = NIMA(base_model)
attack_model.load_state_dict(torch.load('/home/luning/PycharmProjects/mac_code/luning_experimental_code/NIMA/epoch-82.pth'))
attack_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target = 3
target = Variable(torch.Tensor([float(target)]).to(device).long())
image_path = "/home/luning/PycharmProjects/mac_code/luning_experimental_code/NIMA/test_images/kodim11.png"
#得到正确的NIMA的得分
score = 0.0
images = Image.open(image_path).convert('RGB')
images = img_transform(images)
images = images.unsqueeze(dim=0)
attack = PGD(attack_model, eps=8/255, alpha=1/255, steps=200, random_start=True)
adv_images = attack(images, target)
adv_images_test = adv_images
adv_images = adv_images.squeeze(dim=0)
adv_images = adv_images.cpu().detach().numpy()
adv_images = np.transpose(adv_images, (1, 2, 0))
images = images.squeeze(dim=0)
images = images.numpy()
images = np.transpose(images, (1, 2, 0))
show_images_diffrence(images, adv_images)
img_psnr = psnr(images, adv_images)
imageio.imsave('/home/luning/PycharmProjects/mac_code/luning_experimental_code/experiment_picture/PGD/target_1/kidom24_new.png', images)
imageio.imsave('/home/luning/PycharmProjects/mac_code/luning_experimental_code/experiment_picture/PGD/target_1/nima_kidom24_new.png', adv_images)
imageio.imsave('/home/luning/PycharmProjects/mac_code/luning_experimental_code/experiment_picture/PGD/target_1/difference_24.png', (adv_images-images)/255)
outputs_1 = attack_model(adv_images_test)
outputs_1 = outputs_1.view(10, 1)
scores = 0.0
for j, e in enumerate(outputs_1, 1):
    scores += j * e
print('scores = ', scores)
# imageio.imsave('/home/luning/luning_code/luning/code/experiment_picture/PGD/target_1/diffrence.png', images-adv_images)
print('psnr = ', img_psnr)