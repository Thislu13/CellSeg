import os
import shutil
import json

img_dir = '/media/Data1/user/hedongdong/wqs/00.code/data/train_data/raw_data/train/image'
instancemask_dir = '/media/Data1/user/hedongdong/wqs/00.code/data/train_data/raw_data/train/instancemask'


out_dir = '/media/Data1/user/hedongdong/wqs/00.code/data/train_data/train_0'

for img in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img)
    img_name = img.split('.')[0][:-4]
    print(img_name)

    instancemask = img_name+'-instancemask.tif'
    instancemask_path = os.path.join(instancemask_dir, instancemask)

    raw_file = img_name+'.tif'
    instancemask_file = img_name+'_mask.tif'


    out_raw_path = os.path.join(out_dir, raw_file)
    out_instancemask_path = os.path.join(out_dir, instancemask_file)

    shutil.copy(img_path, out_raw_path)
    shutil.copy(instancemask_path, out_instancemask_path)

