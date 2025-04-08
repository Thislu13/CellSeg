import os
import shutil
import json


json_path = '/media/Data1/user/hedongdong/wqs/00.code/data/data_json_3_20.json'
raw_dir = '/media/Data1/user/hedongdong/wqs/00.code/data/raw/'
# pro_dir = '/media/Data1/user/hedongdong/wqs/00.code/data/pro_label/'
pro_dir = '/media/Data1/user/hedongdong/wqs/00.code/data/pro/test/masks/'
with open(json_path, 'r') as f:
    json_data = json.load(f)

num = 0

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


for key,value in json_data.items():
    dir = value['dir']
    if value['train'] == 0:
        to_path = os.path.join(pro_dir,value['name']+'-mask.tif')


        raw_path = os.path.join(value['path'],value['name']+'-mask.tif')

        shutil.copy(raw_path,to_path)
        num += 1
        print(num)

