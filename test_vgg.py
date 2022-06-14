import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import NIMA.main
import vgg

if __name__ == '__main__':
    NIMA.main.print_hi('123')
    img = Image.open('/home/luning/luning_code/luning/code/picture/trian/0.jpg')
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    print('img_type =', type(img))
    print('img_shape', img.shape)
    a = vgg.FeatureVisualization(img, 2).show_feature_to_img()
    plt.imshow(a)
    plt.show()
