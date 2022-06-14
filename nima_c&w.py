import torch.optim as optim
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
img_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
    transforms.ToTensor()
])
def psnr(target, ref):
    # target:目标图像  ref:参考图像  scale:尺寸大小
    diff = ref - target
    diff = diff.flatten('C')
    mse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(1.0 / mse)
def show_images_diffrence(img, nima_img):
    plt.title('Original_img')
    plt.imshow(img)
    # plt.savefig('/home/luning/luning_code/luning/code/experiment_picture/PGD/target_1/orig_img_2.png')
    plt.show()

    plt.title('pgd_img')
    plt.imshow(nima_img)
    # plt.savefig('/home/luning/luning_code/luning/code/experiment_picture/PGD/target_1/nima_img_2.png')
    plt.show()

    plt.title('diffrence')
    plt.imshow(img-nima_img)
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

class CW(Attack):
    r"""
    CW in the paper 'Towards Evaluating the Robustness of Neural Networks'
    [https://arxiv.org/abs/1608.04644]
    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        c (float): c in the paper. parameter for box-constraint. (DEFAULT: 1e-4)
            :math:`minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))`
        kappa (float): kappa (also written as 'confidence') in the paper. (DEFAULT: 0)
            :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
        steps (int): number of steps. (DEFAULT: 1000)
        lr (float): learning rate of the Adam optimizer. (DEFAULT: 0.01)

    .. warning:: With default c, you can't easily get adversarial images. Set higher c like 1.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.CW(model, c=1, kappa=0, steps=1000, lr=0.01)
        >>> adv_images = attack(images, labels)

    .. note:: Binary search for c is NOT IMPLEMENTED methods in the paper due to time consuming.

    """

    def __init__(self, model, c=1e-4, kappa=0, steps=1000, lr=0.01):
        super(CW, self).__init__("CW", model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)

        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        optimizer = optim.Adam([w], lr=self.lr)

        for step in range(self.steps):
            score = 0.0
            # Get Adversarial Images
            adv_images = self.tanh_space(w)
            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()
            outputs = self.model(adv_images)
            outputs = outputs.view(10, 1)
            for j, e in enumerate(outputs, 1):
                score += j * e
            f_loss = self.f(score, labels).sum()
            cost = L2_loss + self.c * f_loss
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            # Update Adversarial Images
            _, pre = torch.max(score.detach(), 1)
            correct = (pre == labels).float()

            mask = (1 - correct) * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            # Early Stop when loss does not converge.
            if step % (self.steps // 10) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        return best_adv_images

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x * 2 - 1)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)
        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())

        return torch.clamp(self._targeted * (i - j), min=-self.kappa)

base_model = models.vgg16(pretrained=True)
attack_model = NIMA(base_model)
attack_model.load_state_dict(torch.load('/home/luning/PycharmProjects/mac_code/luning_experimental_code/NIMA/epoch-82.pth'))
attack_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target = 9
target = Variable(torch.Tensor([float(target)]).to(device).long())
image_path = "/home/luning/PycharmProjects/mac_code/luning_experimental_code/experiment_picture/FGSM/target_1/bird.png"
#得到正确的NIMA的得分
score = 0.0
images = Image.open(image_path).convert('RGB')
images = img_transform(images)
images = images.unsqueeze(dim=0)
attack = CW(attack_model, c=1, kappa=0, steps=1000, lr=0.01)
adv_images = attack(images, target)
adv_images = adv_images.squeeze(dim=0)
adv_images = adv_images.cpu().detach().numpy()
adv_images = np.transpose(adv_images, (1, 2, 0))
images = images.squeeze(dim=0)
images = images.numpy()
images = np.transpose(images, (1, 2, 0))
show_images_diffrence(images, adv_images)
img_psnr = psnr(images, adv_images)
print('psnr = ', img_psnr)