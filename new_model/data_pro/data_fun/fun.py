import os
import glob
import cv2
import numpy as np
from PIL import Image
import tifffile as tiff



class CellSegDataset:
    def __init__(self, img_dir, label_dir):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, '*.tif')))
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, '*.tif')))
        print(len(self.img_paths))
        print(len(self.label_paths))
        assert len(self.img_paths) == len(self.label_paths), "The number of images does not match the number of labels."


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        file_name = self.img_paths[idx].split('/')[-1].split('.')[0][:-4]



        img = np.array(Image.open(self.img_paths[idx]))
        if len(img.shape) == 2:
            img = np.stack((img, img, img), axis=-1)

        label = np.array(Image.open(self.label_paths[idx]))
        if len(label.shape) > 2:
            label = label[..., 0]

        if label.dtype != np.uint16:
            print(f'label dtype {label.dtype}')
        if img.dtype != np.uint16  and img.dtype != np.uint8:
            print(f'img dtype {img.dtype}')

        return {
            'name': file_name,
            "img": img,
            "label": label
        }


def save_image(img, save_path):
    """
    安全保存处理后的图像(3通道)
    参数:
        img: np.ndarray (H,W,3) uint8或uint16
        save_path: 保存路径
    """
    print(img.shape)

    if img.dtype == np.uint8:
        tiff.imsave(save_path, img)
    elif img.dtype == np.uint16:
        tiff.imsave(save_path, img)
    else:
        raise ValueError(f"不支持的图像数据类型: {img.dtype}")


def save_label(label, save_path):
    """
    安全保存处理后的标签(单通道)
    参数:
        label: np.ndarray (H,W) uint16
        save_path: 保存路径
    """
    # 确保是单通道
    print(label.shape)
    assert label.ndim == 2, "标签应该是单通道"

    if label.dtype == np.uint16:
        tiff.imsave(save_path, label)
    else:
        raise ValueError("标签应该是uint16类型")