import os
import cv2
import numpy as np
# import tifffile as tiff


mask_dir = '/media/Data1/user/hedongdong/wqs/00.code/data/out/old'
raw_dir = '/media/Data1/user/hedongdong/wqs/00.code/data/pro/test/images'
out_dir = '/media/Data1/user/hedongdong/wqs/00.code/data/out/merge'


for stain_type in os.listdir(mask_dir):
    stain_type_dir = os.path.join(mask_dir, stain_type)
    merge_stain_type_dir = os.path.join(out_dir, stain_type)
    os.makedirs(merge_stain_type_dir, exist_ok=True)
    for model  in os.listdir(stain_type_dir):
        # if model == 'MEDIAR': continue
        print(stain_type, model)
        model_dir = os.path.join(stain_type_dir, model)
        merge_model_dir = os.path.join(merge_stain_type_dir, model)
        os.makedirs(merge_model_dir, exist_ok=True)
        for img in os.listdir(model_dir):
            img_mask_path = os.path.join(model_dir, img)
            img_raw_path = os.path.join(raw_dir, img)
            img_merge_path = os.path.join(merge_model_dir, img)

            print(img_mask_path, img_raw_path, img_merge_path)

            mask_img = cv2.imread(img_mask_path, cv2.IMREAD_ANYDEPTH)
            raw_img = cv2.imread(img_raw_path)

            if mask_img is None:
                mask_img = cv2.imread(img_mask_path)
            if mask_img is None:
                raise ValueError("图像加载失败，请检查文件格式是否支持32位")

            mask_img = mask_img.astype(np.uint8)
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

            if len(raw_img)==2:
                raw_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
            mask_img[mask_img>0] = 1
            r_layer = np.zeros_like(raw_img)
            r_layer[:, :, 2] = 255


            # 转换为float避免溢出
            raw_float = raw_img.astype(np.float32)
            r_float = r_layer.astype(np.float32)

            # 应用掩膜混合
            # 使用np.where处理更高效
            blended = np.where(
                mask_img[..., None].astype(bool),  # 增加维度用于广播
                raw_float*0.7 + r_float*0.3,
                raw_float
            ).astype(np.uint8)

            cv2.imwrite(img_merge_path, blended)
            print(f"结果已保存至：{img_merge_path}")
            # exit()
