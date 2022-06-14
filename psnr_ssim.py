import math
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
img_transform = transforms.Compose([
    transforms.ToTensor()
])
def psnr(target, ref):
    # target:目标图像  ref:参考图像  scale:尺寸大小
    diff = ref - target
    diff = diff.flatten('C')
    mse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(1.0 / mse)
orig_image_path = '/home/luning/luning_code/luning/code/experiment_picture/FGSM/target_1/nima_img.png'
adv_image_path = '/home/luning/luning_code/luning/code/experiment_picture/FGSM/target_1/orig_img.png'
orig_img = Image.open(orig_image_path).convert('RGB')
orig_img = img_transform(orig_img).numpy()
nima_img = Image.open(adv_image_path).convert('RGB')
nima_img = img_transform(nima_img).numpy()
img_psnr = psnr(nima_img, orig_img)
print('psnr = ', img_psnr)