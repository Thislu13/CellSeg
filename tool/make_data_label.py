import os
import random
import json


dir = '/media/Data1/user/hedongdong/wqs/00.code/data/raw/'
json_path = '/media/Data1/user/hedongdong/wqs/00.code/data/data_json.json'

json_data = {}

for filename in os.listdir(dir):
    sn_path = os.path.join(dir, filename)
    if not os.path.isdir(sn_path):
        continue

    sn_list = os.listdir(sn_path)
    split_index = int(len(sn_list)*0.8)
    random.shuffle(sn_list)
    train_files = sn_list[:split_index]
    test_files = sn_list[split_index:]
    print(filename)
    print(len(test_files),len(train_files))

    for sn_name in train_files + test_files:
        file_path = os.path.join(sn_path, sn_name)

        if sn_name in json_data:
            print(file_path)


        json_data.setdefault(sn_name, {
            'name': sn_name,
            'path': file_path,
            'dir': filename,
            'train': 1 if sn_name in train_files else 0  # 使用更高效的判断
        })



# 写入JSON文件（添加实际保存功能）
with open(json_path, 'w') as f:
    json.dump(json_data, f, indent=2)

print(f"处理完成！共处理 {len(json_data)} 个文件，结果已保存至 {json_path}")