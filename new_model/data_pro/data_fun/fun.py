import cv2
import numpy as np
import copy


class BoundaryExclusion:
    def __init__(self, min_cell_area=196, border_width=2):
        self.min_cell_area = min_cell_area
        self.border_width = border_width

    def __call__(self, data):
        label = data["label"].copy()
        original = label.copy()

        # 1. 腐蚀法获取内边界
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(label.astype(np.uint8), kernel)
        boundary = (label - eroded).astype(bool)  # 得到真实内边界

        # 2. 标记小细胞区域
        cell_ids, counts = np.unique(original, return_counts=True)
        small_cell_mask = np.zeros_like(label, dtype=bool)
        for cid, cnt in zip(cell_ids, counts):
            if cid != 0 and cnt < self.min_cell_area:
                small_cell_mask |= (original == cid)

        # 3. 排除边界（保留小细胞）
        label[boundary & ~small_cell_mask] = 0

        # 4. 处理图像物理边缘
        h, w = label.shape[-2:]
        valid_region = np.zeros_like(label, dtype=bool)
        valid_region[...,
        self.border_width:h - self.border_width,
        self.border_width:w - self.border_width] = True
        label = np.where(valid_region, label, original)

        data["label"] = label
        return data


class IntensityDiversification:
    """对随机选择的细胞区域进行强度变换"""

    def __init__(self, change_ratio=0.4, scale_range=(0.6, 1.4)):
        self.change_ratio = change_ratio
        self.scale_range = scale_range

    def __call__(self, data):
        img = data["img"].copy().astype(np.float32)
        label = data["label"].copy()

        # 获取有效细胞ID
        cell_ids = np.unique(label)
        cell_ids = cell_ids[cell_ids != 0]
        if len(cell_ids) == 0:
            return data

        # 随机选择细胞
        n_select = max(1, int(len(cell_ids) * self.change_ratio))
        selected = np.random.choice(cell_ids, n_select, replace=False)

        # 创建掩膜并生成随机因子
        mask = np.isin(label, selected)
        scale = np.random.uniform(*self.scale_range)

        # 应用强度变换
        img_transformed = img * (1 - mask) + (img * mask * scale)
        img_transformed = np.clip(img_transformed, 0, 255).astype(np.uint8)

        data["img"] = img_transformed
        return data