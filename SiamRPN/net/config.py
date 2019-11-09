import numpy as np


class Config:
    instance_size = 271
    exemplar_size = 127
    instance_crop_size = 511
    context_margin_amount = 0.5
    total_stride = 8
    pairs_per_sequence_per_epoch = 2
    anchor_scales = np.array([8, ])
    anchor_ratio = np.array([0.33, 0.5, 1, 2, 3])
    scale_range = (0.001, 0.7)  # what's this
    anchor_num = len(anchor_ratio) * len(anchor_ratio)
    anchor_base_size = 8
    score_map_size = (instance_size - exemplar_size) / total_stride + 1
    scale_range = (0.001, 0.7)
    ratio_range = (0.1, 10)
    frame_range = 100
    sample_type = 'uniform'
    gray_ratio = 0.25
    scale_resize = 0.15  # 训练时对instance_img的缩放
    max_shift = 12       # 训练时对中心的最大偏移量
    exemplar_stretch = False
