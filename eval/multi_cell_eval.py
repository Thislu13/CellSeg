import glob
import tqdm
import os
import json


import logging
models_logger = logging.getLogger(__name__)

import numpy as np
import tifffile
import cv2 as cv
from skimage.measure import label
from skimage import io
from collections import OrderedDict
from metrics import Metrics
import argparse
import pandas as pd
import subprocess

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


