import torch
import torch.nn as nn
from glob import glob
from PIL import Image
class Dataset(nn.Module):
    def __init__(self, file, image_transform=None):
        super(Dataset, self).__init__()
        self.file = file
        self.transform = image_transform

    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        subfile = self.file[index]
        image = Image.open(subfile)
        if self.transform != None:
            image = self.transform(image)
        image = torch.tensor(image)
        return image

if __name__ == '__main__':
    img_path = '/home/luning/luning_code/luning/code/picture/trian'
    image = glob(img_path + '/*' + '.jpg')
    image = Dataset(image)[0]
    print(image)
