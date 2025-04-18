import os
import shutil
import json


json_path = '/media/Data1/user/hedongdong/wqs/00.code/data/data_json_3_20.json'
raw_dir = '/media/Data1/user/hedongdong/wqs/00.code/data/raw/'
# pro_dir = '/media/Data1/user/hedongdong/wqs/00.code/data/pro_label/'
out_dir = '/media/Data1/user/hedongdong/wqs/00.code/data/train_data/raw_data'
with open(json_path, 'r') as f:
    json_data = json.load(f)


# for key,value in json_data.items():
#     dir = value['dir']
#     if value['train'] == 1:
#         to_dir = os.path.join(pro_dir,'train')
#         to_dir = os.path.join(to_dir,dir)
#         os.makedirs(to_dir, exist_ok=True)
#         to_path = os.path.join(to_dir,value['name']+'-img.tif')
#     elif value['train'] == 0:
#         to_dir = os.path.join(pro_dir,'test')
#         to_dir = os.path.join(to_dir, dir)
#         os.makedirs(to_dir, exist_ok=True)
#         to_path = os.path.join(to_dir,value['name']+'-img.tif')
#
#     raw_path = os.path.join(value['path'],value['name']+'-mask.tif')
#
#     shutil.copy(raw_path,to_path)
#     num += 1
#     print(num)

num_test = 0
num_train = 0
for key,value in json_data.items():
    # dir = value['dir']
    if value['train'] == 1:
        # img
        img_raw_path  = os.path.join(value['path'],f"{value['name']}-img.tif")
        img_to_path = os.path.join(out_dir, 'train', 'image', f"{value['name']}-img.tif")
        shutil.copy(img_raw_path, img_to_path)
        #instancemask
        instancemask_raw_path = os.path.join(value['path'],f"{value['name']}-instancemask.tif")
        instancemask_to_path = os.path.join(out_dir, 'train', 'instancemask', f"{value['name']}-instancemask.tif")
        shutil.copy(instancemask_raw_path, instancemask_to_path)
        #mask
        mask_raw_path = os.path.join(value['path'],f"{value['name']}-mask.tif")
        mask_to_path = os.path.join(out_dir, 'train', 'mask', f"{value['name']}-mask.tif")
        shutil.copy(mask_raw_path, mask_to_path)
        num_train += 1
    else:
        # img
        img_raw_path  = os.path.join(value['path'],f"{value['name']}-img.tif")
        img_to_path = os.path.join(out_dir, 'test', 'image', f"{value['name']}-img.tif")
        shutil.copy(img_raw_path, img_to_path)
        #instancemask
        instancemask_raw_path = os.path.join(value['path'],f"{value['name']}-instancemask.tif")
        instancemask_to_path = os.path.join(out_dir, 'test', 'instancemask', f"{value['name']}-instancemask.tif")
        shutil.copy(instancemask_raw_path, instancemask_to_path)
        #mask
        mask_raw_path = os.path.join(value['path'],f"{value['name']}-mask.tif")
        mask_to_path = os.path.join(out_dir, 'test', 'mask', f"{value['name']}-mask.tif")
        shutil.copy(mask_raw_path, mask_to_path)
        num_test += 1



print(num_train)
print(num_test)
