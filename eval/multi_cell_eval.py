import glob
from tqdm import tqdm
import os
import json


import logging
models_logger = logging.getLogger(__name__)

import numpy as np
import tifffile as tiff
import cv2 as cv
from skimage.measure import label
from skimage import io
from collections import OrderedDict
from metrics import Metrics
import argparse
import pandas as pd
import subprocess





def search_files(file_path, exts):

    file_list = list()

    for root, dirs, files in os.walk(file_path):
        if len(files) == 0:
            continue
        for file in files:
            filename, ext = os.path.splitext(file)
            if ext in exts: file_list.append(os.path.join(root, file))

    return file_list




class CellSegEval:
    def __init__(self, method: str = None):
        self._method = method
        self._gt_list = list()
        self._dt_list = list()
        self._object_metrics = None
        self._suitable_shape = None

    def set_method(self, method: str):
        self._method = method

    def _load_iamge(self, image_path: str):
        arr_ = np.zeros(self._suitable_shape, dtype=np.uint8)
        arr = tifffile.imread(image_path,key=0) # 避免图像被当作多维数组读出
        h, w = arr.shape
        arr_[:h, :w] = arr
        arr_ = label(arr_, connectivity=2) # 采用八通联

        return arr_

    def evaluate(self, gt_path: str, dt_path: str):

        for i in [gt_path, dt_path]:
            assert os.path.exists(i), 'Path does not exist: {}'.format(i)
        print(f'gt_path: {gt_path}\ndt_path: {dt_path}')

        # 读取gt dt列表
        if os.path.isfile(gt_path):
            self._gt_list = [gt_path]
        else:
            img_list_gt = search_files(gt_path, ['.tif', ',png', '.jpg', '.png'])
            self._gt_list = [image_path for image_path in img_list_gt if 'mask' in image_path] # 读 包含mask图片

        if os.path.isfile(dt_path):
            self._dt_list = [dt_path]
        else:
            self._dt_list = search_files(dt_path, ['.tif', ',png', '.jpg', '.png'])
        self._gt_list = [image_path for image_path in self._dt_list if image_path.replace('mask', 'img').replace(gt_path, dt_path) in self._dt_list] #只读DT中有对应GT的图片
        assert len(self._gt_list) == len(self._dt_list), 'Length of list GT {} are not equal to DT {}'.format(len(self._gt_list), len(self._dt_list))

        # 统计图片形状
        gt_arr = list()
        dt_arr = list()
        shape_list = list()
        for i in self._dt_list:
            dt = tiff.imread(i,key=0)
            shape_list.append(dt.shape)
        w = np.max(np.array(shape_list)[:, 1])
        h = np.max(np.array(shape_list)[:, 0])
        self._suitable_shape = (h, w)
        models_logger.info('Uniform size {} into {}'.format(list(set(shape_list)), self._suitable_shape))

        # 加载数据
        for img_file in tqdm(self._dt_list, desc='Load data {}'.format(self._method)):
            gt_img = self._load_iamge(image_path=img_file.replace('img', 'mask').replace(dt_path, gt_path))
            dt_img = self._load_iamge(image_path=img_file)
            assert gt_img.shape == dt_img.shape, 'GT_img and DT_img shapes do not match'
            gt_arr.append(gt_img)
            dt_arr.append(dt_img)
        gt_arr = np.array(gt_arr)
        dt_arr = np.array(dt_arr)





def main(args, para):

    dataset_name = os.path.basename(os.path.dirname(args.gt_path))
    print(f'dataset_name: {dataset_name}')

    visible_folders = [folder for folder in os.listdir(args.dt_path) if not folder.startswith('.')]
    models = visible_folders
    print(f'models: {models}')

    dct = {}
    # gt_path = os.path.join(args.gt_path)
    gt_path = args.gt_path
    dataset_dct = {}

    for model in models:
        dt_path = os.path.join(args.dt_path, model)








usage = """ Evaluate cell segmentation """

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("-g", "--gt_path", action="store", dest="gt_path", type=str, required=True,
                        help="Input GT path.")
    parser.add_argument("-d", "--dt_path", action="store", dest="dt_path", type=str, required=True,
                        help="Input DT path.")
    parser.add_argument("-o", "--output_path", action="store", dest="output_path", type=str, required=True,
                        help="Output result path.")
    parser.set_defaults(func=main)

    (args, para) = parser.parse_known_args()
    print(args, para)
    para.func(args, para)


