import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models, metrics, train
from glob import glob
from pathlib import Path


def check_masks(mask_dir):
    mask_dir = Path(mask_dir)
    mask_files = list(mask_dir.glob("*_masks.*"))  # 根据实际后缀调整

    for mask_file in mask_files:
        mask = io.imread(mask_file)
        num_cells = len(np.unique(mask)) - 1  # 排除背景0
        print(f"{mask_file.name}: 细胞数量 = {num_cells}")
        if num_cells < 1:
            print(f"  警告：{mask_file.name} 无有效细胞！")
train_dir = '/media/Data1/user/hedongdong/wqs/00.code/data/train_data/train_fine'

pretrained_model = '/media/Data1/user/hedongdong/wqs/00.code/data/weight/cellpose/models/cellpose_all_1000_0.01'
model_path = '/media/Data1/user/hedongdong/wqs/00.code/data/weight/cellpose'

# check_masks(train_dir)


use_GPU = core.use_gpu()


# initial_model = 'cyto'
epochs = 1000
Use_Default_Advanced_Parameters = True
learning_rate = 0.01
weight_decay = 0.0001

# start logger (to see training across epochs)
logger = io.logger_setup()

# DEFINE CELLPOSE MODEL (without size model)
# model = models.CellposeModel(gpu=use_GPU, model_type=initial_model)
model = models.CellposeModel(gpu=True, pretrained_model=pretrained_model)

# get files
output = io.load_train_test_data(train_dir, mask_filter = '_mask')
train_data, train_labels, _, _, _, _= output

print(len(train_data))
print(len(train_labels))
out = train.train_seg(model.net,
                              train_data=train_data,
                              train_labels=train_labels,
                              batch_size=32,
                              learning_rate=learning_rate,
                              n_epochs=epochs,
                              weight_decay=weight_decay,
                              SGD=True,
                            channels = [0,0],
                            save_path=model_path,
)

# print(new_model_path)