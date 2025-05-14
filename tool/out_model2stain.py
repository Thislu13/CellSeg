import os
import json
import shutil

model_dir = '/media/Data1/user/hedongdong/wqs/00.code/data/out/new/model'
stain_dir = '/media/Data1/user/hedongdong/wqs/00.code/data/out/new/stain'
json_path = '/media/Data1/user/hedongdong/wqs/00.code/data/data_json_3_20.json'

json_data = json.load(open(json_path))


for model in os.listdir(model_dir):
    print(model)
    for image in os.listdir(os.path.join(model_dir, model)):
        image_name = image[:-8]
        stain = json_data[image_name]["dir"]

        stain_path = os.path.join(stain_dir, stain)
        model_path = os.path.join(stain_path, model)

        image_raw_path = os.path.join(model_dir, model, image)
        image_to_path = os.path.join(model_path, image)

        os.makedirs(stain_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

        shutil.copy(image_raw_path, image_to_path)

        # exit()