import cv2
import numpy as np
import math
from PIL import Image
from torchvision import transforms
# def ssim(img1, img2):
#     C1 = (0.01 * 255) ** 2
#     C2 = (0.03 * 255) ** 2
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())
#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1 ** 2
#     mu2_sq = mu2 ** 2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                             (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()
#
#
# def calculate_ssim(img1, img2):
#     '''calculate SSIM
#     the same outputs as MATLAB's
#     img1, img2: [0, 255]
#     '''
#     if not img1.shape == img2.shape:
#         raise ValueError('Input images must have the same dimensions.')
#     if img1.ndim == 2:
#         return ssim(img1, img2)
#     elif img1.ndim == 3:
#         if img1.shape[2] == 3:
#             ssims = []
#             for i in range(3):
#                 ssims.append(ssim(img1, img2))
#             return np.array(ssims).mean()
#         elif img1.shape[2] == 1:
#             return ssim(np.squeeze(img1), np.squeeze(img2))
#     else:
#         raise ValueError('Wrong input image dimensions.')
# img1 = cv2.imread("/home/luning/luning_code/luning/code/experiment_picture/PGD/target_1/nima_img_2.png", 0)
# img2 = cv2.imread("/home/luning/luning_code/luning/code/experiment_picture/PGD/target_1/orig_img_2.png", 0)
# ss = calculate_ssim(img1, img2)
# print(ss)
img_transform = transforms.Compose([
    transforms.ToTensor()
])
def psnr(target, ref):
    # target:目标图像  ref:参考图像  scale:尺寸大小
    diff = ref - target
    diff = diff.flatten('C')
    mse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(1.0 / mse)
image_path_1 = '/home/luning/PycharmProjects/mac_code/luning_experimental_code/experiment_picture/STN/fgsm_orig_bird.png'
image_path_2 = '/home/luning/PycharmProjects/mac_code/luning_experimental_code/experiment_picture/STN/fgsm_nima_bird_1.png'
img_1 = Image.open(image_path_1).convert('RGB')
img_1 = img_transform(img_1)
img_2 = Image.open(image_path_2).convert('RGB')
img_2 = img_transform(img_2)
img_1 = img_1.numpy()
img_1 = img_1.transpose(1, 2, 0)
img_2 = img_2.numpy()
img_2 = img_2.transpose(1, 2, 0)
img_psnr = psnr(img_1, img_2)
print('img_psnr = ', img_psnr)