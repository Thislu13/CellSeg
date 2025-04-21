import cv2
import numpy as np
import copy


class BoundaryExclusion:
    """
    边界标签优化 减少边界干扰
    boundary_processor = BoundaryExclusion(min_cell_area=196, border_width=2)
    boundary_processor(data: dict)
    {
    'img': np.ndarray (H,W,3) uint8 or uint16
    'label': np.ndarray (H,W) uint16
    }
    """

    def __init__(self, min_cell_area=196, border_width=2):
        self.min_cell_area = min_cell_area
        self.border_width = border_width

    def __call__(self, data):
        label = data["label"].copy()
        original = label.copy()
        boundary = np.zeros(label.shape, dtype=bool)
        # 1. 腐蚀法获取内边界
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(label.astype(np.uint16), kernel)
        boundary[(eroded == 0) & (original != 0)] = 1   # 得到真实内边界

        # 2. 标记小细胞区域
        cell_ids, counts = np.unique(original, return_counts=True)
        small_cell_mask = np.zeros_like(label, dtype=bool)
        for cid, cnt in zip(cell_ids, counts):
            if cid != 0 and cnt < self.min_cell_area:
                small_cell_mask |= (original == cid)

        # 3. 排除边界（保留小细胞）
        label[boundary & ~small_cell_mask] = 0

        # 4. 处理图像物理边缘
        h, w = label.shape
        valid_region = np.zeros_like(label, dtype=bool)
        valid_region[self.border_width:h - self.border_width,
        self.border_width:w - self.border_width] = True
        label = np.where(valid_region, label, original)


        data["label"] = label


        return data


class IntensityDiversification:
    """
    对随机选择的细胞区域进行强度变换
    intensity_processor = IntensityDiversification(change_ratio=0.4, scale_range=(0.6, 1.4))
    intensity_processor(data: dict)
    {
    'img': np.ndarray (H,W,3) uint8 or uint16
    'label': np.ndarray (H,W) uint16
    }
    """

    def __init__(self, change_ratio=0.4, scale_range=(0.6, 1.4)):
        self.change_ratio = change_ratio
        self.scale_range = scale_range

    def __call__(self, data):
        img = data["img"].copy()
        label = data["label"].copy()

        # 保存原始类型和值范围
        is_8bit = img.dtype == np.uint8
        original_dtype = img.dtype
        if (original_dtype == np.uint16  or original_dtype == np.uint8):
            print('1111111111')
        # # 转换为float32处理
        # img_float = img.astype(np.float32)

        # 获取有效细胞ID
        cell_ids = np.unique(label)
        cell_ids = cell_ids[cell_ids != 0]
        if len(cell_ids) == 0:
            return data

        # 随机选择细胞
        n_select = max(1, int(len(cell_ids) * self.change_ratio))
        selected = np.random.choice(cell_ids, n_select, replace=False)

        # 创建3D掩膜 (H,W,1) -> (H,W,3)
        mask = np.isin(label, selected)
        mask = np.expand_dims(mask, axis=-1)  # 增加通道维度
        mask = np.repeat(mask, 3, axis=-1)  # 复制到3个通道

        # 生成随机缩放因子（所有通道相同或不同）
        if np.random.rand() > 0.5:
            # 所有通道相同变换
            scale = np.random.uniform(*self.scale_range)
        else:
            # 各通道独立变换
            scale = np.random.uniform(*self.scale_range, size=3)
            scale = np.reshape(scale, (1, 1, 3))  # 适合广播的形状

        # 应用强度变换
        img_transformed = img * (1 - mask) + (img * mask * scale)

        # 转换回原始数据类型
        if is_8bit:
            img_save = np.clip(img_transformed, 0, 255).astype(np.uint8)
        else:
            img_save = np.clip(img_transformed, 0, 65535).astype(np.uint16)

        data["img"] = img_save
        return data