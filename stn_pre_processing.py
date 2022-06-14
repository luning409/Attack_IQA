import torch
import numpy as np
import stn_model
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = stn_model.Net().to(device)


# 分成32*32的块
def split_32(img):
    rgb_list = []  # 3个通道里的值
    for i in range(3):
        block_list = []
        for r in range(57):
            for c in range(77):
                block_list.append(img[i][r * 8:r * 8 + 32, c * 8:c * 8 + 32])
        rgb_list.append(block_list)
    return rgb_list
 

# 丢进STN
def stn_process(rgb_list):
    after_rgb_list = []  # 处理过后的3个通道里的值
    # 丢进STN
    for i in range(len(rgb_list)):
        after_block_list = []
        for r in range(len(rgb_list[i])):
            rgb_tensor = rgb_list[i][r]
            rgb_tensor = torch.unsqueeze(rgb_tensor, dim=0)
            rgb_tensor = torch.unsqueeze(rgb_tensor, dim=0)
            rgb_tensor = rgb_tensor.to(device).float()
            with torch.no_grad():
                output = model(rgb_tensor)
            after_block_list.append(output[0][0])
        after_rgb_list.append(after_block_list)
    return after_rgb_list


# 取出32*32块的中间的8*8的块
def extract_8(after_rgb_list):
    rgb_center_block = []  # 3个通道的8*8的块
    for i in range(len(after_rgb_list)):
        block = after_rgb_list[i][:]
        center_block = []
        for j in range(len(block)):
            center_block.append(block[j][12:20, 12:20])
        rgb_center_block.append(center_block)
    return rgb_center_block


# 组合8*8的块
def compose_8(rgb_center_block):
    img_compose = []  # 最终组合好的图像
    for i in range(len(rgb_center_block)):
        temp_list = rgb_center_block[i]  # 里面放着4398个8*8的矩阵
        width_list = []
        height_list = []
        for j in range(57):
            width_list.append(temp_list[j * 77:(j + 1) * 77])
        for j in range(57):
            temp_var = width_list[j][0]
            for k in range(76):
                temp_var = torch.cat((temp_var, width_list[j][k + 1]), 1)
            height_list.append(temp_var)
        temp_values = height_list[0]
        for m in range(56):
            temp_values = torch.cat((temp_values, height_list[m]), 0)
        img_compose.append(temp_values)
    return img_compose


# 用原始图像的边来填充组合好后的图像
def compose_img(img_compose, img):
    for i in range(3):
        img[i][12:468, 12:628] = img_compose[i][:]
    return img

def process_img_stn(img):
    rgb_list = split_32(img)
    after_rgb_list = stn_process(rgb_list)
    rgb_center_block = extract_8(after_rgb_list)
    img_compose = compose_8(rgb_center_block)
    img_process_after = compose_img(img_compose, img)
    return img_process_after


if __name__ == '__main__':
    height = 480
    weight = 640
    img_mean = torch.tensor([0.485, 0.456, 0.406])
    img_std = torch.tensor([0.229, 0.224, 0.225])
    img_path = '/home/luning/luning_code/luning/code/picture/trian/0.jpg'
    img = Image.open(img_path)
    image_transform = transforms.Compose([
        transforms.Resize((height, weight), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std)
    ])
    img = image_transform(img)
    img_after_stn = process_img_stn(img)
    img_after_stn = np.transpose(img_after_stn, (1, 2, 0))
    img_after_stn = img_after_stn * img_std + img_mean
    plt.imshow(img_after_stn)
    plt.show()
