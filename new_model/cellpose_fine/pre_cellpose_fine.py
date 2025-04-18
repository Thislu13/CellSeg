import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models, metrics, train
from glob import glob
from pathlib import Path


diameter = 0
flow_threshold = 0.4
cellprob_threshold = 0

model_path= '/media/Data1/user/hedongdong/wqs/00.code/data/weight/cellpose/models/cellpose_1744948114.0810616'
dir = '/media/Data1/user/hedongdong/wqs/00.code/data/train_data/raw_data/test/image'

files = io.get_image_files(dir, mask_filter='_mask')
# print(files)
images = [io.imread(f) for f in files]

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

io.save_masks(images,
              masks,
              flows,
              files,
              channels=[0, 0],
              png=True, # save masks as PNGs and save example image
              tif=True, # save masks as TIFFs
              save_txt=False, # save txt outlines for ImageJ
              save_flows=False, # save flows as TIFFs
              save_outlines=False, # save outlines as TIFFs
              save_mpl=True # make matplotlib fig to view (WARNING: SLOW W/ LARGE IMAGES)
              )