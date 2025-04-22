import os
import shutil

dir_1 = '/media/Data1/user/hedongdong/wqs/00.code/data/train_data/train_boundary'
dir_2 = '/media/Data1/user/hedongdong/wqs/00.code/data/train_data/train_intensity'


dir_3 = '/media/Data1/user/hedongdong/wqs/00.code/data/train_data/train_all'


num_0 = 0
num_1 = 0


for label_file in os.listdir(dir_1):
    if '_mask.tif' in label_file:
        old_file = os.path.join(dir_1, label_file)
        new_file = os.path.join(dir_3, label_file)
        shutil.copy(old_file, new_file)
        num_0 += 1

for img_file in os.listdir(dir_2):
    if '_mask.tif' not in img_file:
        old_file = os.path.join(dir_2, img_file)
        new_file = os.path.join(dir_3, img_file)
        shutil.copy(old_file, new_file)
        num_1 += 1

print(num_0, num_1)