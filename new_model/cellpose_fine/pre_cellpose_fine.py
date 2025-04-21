import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models, metrics, train
from glob import glob
from pathlib import Path
import cv2


diameter = 0
flow_threshold = 0.4
cellprob_threshold = 0

model_path= '/media/Data1/user/hedongdong/wqs/00.code/data/weight/cellpose/models/cellpose_1000'
dir = '/media/Data1/user/hedongdong/wqs/00.code/data/train_data/raw_data/test/image'
out_dir = '/media/Data1/user/hedongdong/wqs/00.code/data/out/eval/train_1'

files = io.get_image_files(dir, mask_filter='_mask')
# print(files)
images = [io.imread(f) for f in files]
# print(len(files))
# print(files[0])
# exit()



model = models.CellposeModel(gpu=True, pretrained_model=model_path)

diameter = model.diam_labels if diameter == 0 else diameter

masks, flows, styles = model.eval(
    images,
    channels=[0,0],
    diameter=diameter,
    flow_threshold=flow_threshold,
    cellprob_threshold=cellprob_threshold
)


print(len(masks))
print(masks[0].shape)

for i in range(len(masks)):
    mask = masks[i]
    file = files[i].split('/')[-1]
    # print(file)

    save_path = os.path.join(out_dir, file)


    save_mask = np.zeros_like(mask).astype(np.uint8)
    save_mask[mask > 0] = 255

    print(save_path)
    cv2.imwrite(save_path, save_mask)


# io.save_masks(images,
#               masks,
#               flows,
#               files,
#               channels=[0, 0],
#               png=True, # save masks as PNGs and save example image
#               tif=True, # save masks as TIFFs
#               save_txt=False, # save txt outlines for ImageJ
#               save_flows=False, # save flows as TIFFs
#               save_outlines=False, # save outlines as TIFFs
#               save_mpl=True # make matplotlib fig to view (WARNING: SLOW W/ LARGE IMAGES)
#               )