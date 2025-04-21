import fun, utiles
import os
import numpy as np


img_dir = '/media/Data1/user/hedongdong/wqs/00.code/data/train_data/raw_data/train/image'
label_dir = '/media/Data1/user/hedongdong/wqs/00.code/data/train_data/raw_data/train/instancemask'

save_dir = '/media/Data1/user/hedongdong/wqs/00.code/data/train_data/train_boundary'


dataset = fun.CellSegDataset(
    img_dir=img_dir,
    label_dir=label_dir,
)

boundary_processor = utiles.BoundaryExclusion(min_cell_area=64, border_width=2)
interior_processor = utiles.IntensityDiversification(change_ratio=0.4, scale_range=(0.7, 1.4))
print(len(dataset))


for index in range(len(dataset)):
    print(index)
    print(dataset[index]['name'])
    raw = dataset[index]

    img_processed = boundary_processor(dataset[index])
    save_img_path = os.path.join(save_dir, dataset[index]['name']+'.tif')
    save_mask_path = os.path.join(save_dir, dataset[index]['name']+'_mask.tif')



    fun.save_label(img_processed['label'], save_mask_path)
    fun.save_image(img_processed['img'], save_img_path)



