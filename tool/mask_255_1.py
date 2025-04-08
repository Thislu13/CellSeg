import cv2
import os


img_dir = '/media/Data1/user/hedongdong/wqs/00.code/data/pro/train/masks'
mask_dir = '/media/Data1/user/hedongdong/wqs/00.code/data/pro/train_1'



for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    to_path = os.path.join(mask_dir, img_name)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img[img>0] = 1
    cv2.imwrite(to_path, img)