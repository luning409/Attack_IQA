import os
path = r"../luning_experimental_code/dataset"
# 遍历更改文件名
num = 0
path_list = os.listdir(path)
# path_list.sort(key=lambda x: int(x[:-4]))
for file in path_list:
    num = num + 1
    os.rename(os.path.join(path, file), os.path.join(path, str(num))+".png")

