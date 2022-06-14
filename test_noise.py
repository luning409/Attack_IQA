import numpy as np
import random
import cv2
def gaussian_white_noise(intput_signal, mu, sigma):
    '''
    加性高斯白噪声（适用于灰度图）
    :param intput_signal: 输入图像
    :param mu: 均值
    :param sigma: 标准差
    :return:
    '''
    intput_signal_cp = np.copy(intput_signal)  # 输入图像的副本
    m, n = intput_signal_cp[0].shape   # 输入图像尺寸（行、列）
    # 添加高斯白噪声
    for h in range(3):
        for i in range(m):
            for j in range(n):
                intput_signal_cp[h][i, j] = intput_signal_cp[h][i, j] + random.gauss(mu, sigma)
    return intput_signal_cp
img = np.zeros((3, 224, 224))
gaussian_img = gaussian_white_noise(img, 1, 0.1)
gaussian_img *= 255
print('gaussian_img = ', gaussian_img)
gaussian_img = np.transpose(gaussian_img, (1, 2, 0))
gaussian_img = np.clip(gaussian_img, 0, 255)
cv2.imwrite('/home/luning/luning_code/luning/code/test_picture/test_noise.jpg', gaussian_img)
