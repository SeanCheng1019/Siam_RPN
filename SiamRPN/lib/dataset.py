from torch.utils.data.dataset import Dataset
import numpy as np
from net.config import Config
from lib.util import generate_anchors, crop_and_pad, box_delta_in_gt_anchor, compute_iou
from lib.custom_transforms import RandomStretch
from glob import glob
import os
import cv2


class GetDataSet(Dataset):
    """
     之前得分割的训练集和验证集是一个一个视频序列的文件夹，这个类的作用是为了在训练的时候每次迭代时，
     从这些不同序列的文件夹中，得到每个视频序列中一对一对的图片，从而可以对这些图片直接放入网络中去。
     其中有加入一些数据增强的操作。
    """

    def __init__(self, sequence_names, data_dir, z_transforms, x_transforms, meta_data, training=True):
        self.sequence_names = sequence_names
        self.data_dir = data_dir
        self.z_transforms = z_transforms
        self.x_transforms = x_transforms
        self.meta_data = meta_data
        # 初始化的时候，把在一个序列里只出现2帧以内的目标筛除掉。
        self.meta_data = {x[0]: x[1] for x in meta_data}
        # 训练模式都是从一个序列中成对成对的去选的
        self.num = len(sequence_names) if not training else \
            Config.pairs_per_sequence_per_epoch * len(sequence_names)
        for track_sequence_name in self.meta_data.keys():
            track_sequence_info = self.meta_data[track_sequence_name]
            for object_id in list(track_sequence_info.keys()):
                if len(track_sequence_info[object_id]) < 2:
                    del track_sequence_info[object_id]
        self.training = training
        self.max_stretch = Config.scale_resize  # 干啥用
        self.max_shift = Config.max_shift
        self.center_crop_size = Config.exemplar_size  # 裁剪模版用的尺寸
        self.random_crop_size = Config.instance_size  # 裁剪搜索区域用的尺寸
        self.anchors = generate_anchors(Config.total_stride, Config.anchor_base_size, Config.anchor_scales,
                                        Config.anchor_ratio, Config.score_map_size)

    def __getitem__(self, index):  # 何时调用的，何时传入的index参数
        all_idx = np.arange(self.num)
        np.random.shuffle(all_idx)
        # 先选一个序列，然后序列里选个目标，然后选择一帧。
        for idx in all_idx:
            index = index % len(self.sequence_names)
            sequence = self.sequence_names[index]
            trajs = self.meta_data[sequence]
            if len(trajs.keys()) == 0:
                continue
            trkid = np.random.choice(list(trajs.keys()))
            trk_frames = trajs[trkid]
            assert len(trk_frames) > 1, "sequence_name: {}".format(sequence)
            # 选择图片
            exemplar_index = np.random.choice(list(range(len(trk_frames))))
            exemplar_whole_path_name = glob(os.path.join(self.data_dir, sequence, trk_frames[exemplar_index] +
                                                         ".{:02d}.patch*.jpg".format(trkid)))[0]
            exemplar_gt_w, exemplar_gt_h, exemplar_img_w, exemplar_img_h = float(
                exemplar_whole_path_name.split('/')[-1].split('_')[2]), \
                                                                           float(exemplar_whole_path_name.split('/')[
                                                                                     -1].split('_')[4][1:]), \
                                                                           float(exemplar_whole_path_name.split('/')[
                                                                                     -1].split('_')[7]), \
                                                                           float(exemplar_whole_path_name.split('/')[
                                                                                     -1].split('_')[-1][:-4])
            exemplar_ratio = min(exemplar_gt_h / exemplar_gt_w, exemplar_gt_w / exemplar_gt_h)
            exemplar_scale = exemplar_gt_w * exemplar_gt_h / (exemplar_img_w * exemplar_img_h)
            # 为了过滤掉一些特殊的案例
            if not Config.scale_range[0] <= exemplar_scale < Config.scale_range[1]:
                continue
            if not Config.ratio_range[0] <= exemplar_ratio < Config.ratio_range[1]:
                continue
            # 选到模版图片
            exemplar_img = self.imread(exemplar_whole_path_name)
            # 开始选instance image
            frame_range = Config.frame_range
            low_idx = max(0, exemplar_index - frame_range)
            high_idx = min(len(trk_frames), exemplar_index + frame_range)
            # 在low_idx 和 high_idx里去选择一个来作为instance img， 但是为了保证每一个都能平等的选择到，
            # 这里加入一个采样权重来达到这样的目的。
            weights = self.sample_weights(exemplar_index, low_idx, high_idx, Config.sample_type)
            instance_file_name = np.random.choice(trk_frames[low_idx:exemplar_index]
                                                  + trk_frames[exemplar_index + 1:high_idx], p=weights)
            instance_whole_path_name = glob(os.path.join(self.data_dir, sequence, instance_file_name +
                                                         ".{:02d}.patch*.jpg".format(trkid)))[0]
            # 和之前选exemplar的操作一样 （**之后可以考虑这部分重复的代码封装在util的方法里**）
            instance_gt_w, instance_gt_h, instance_img_w, instance_img_h = float(
                instance_whole_path_name.split('/')[-1].split('_')[2]), \
                                                                           float(instance_whole_path_name.split('/')[
                                                                                     -1].split('_')[4][1:]), \
                                                                           float(instance_whole_path_name.split('/')[
                                                                                     -1].split('_')[7]), \
                                                                           float(instance_whole_path_name.split('/')[
                                                                                     -1].split('_')[-1][:-4])
            instance_ratio = min(instance_gt_h / instance_gt_w, instance_gt_w / instance_gt_h)
            instance_scale = instance_gt_w * instance_gt_h / (instance_img_w * instance_img_h)
            # 为了过滤掉一些特殊的案例
            if not Config.scale_range[0] <= instance_scale < Config.scale_range[1]:
                continue
            if not Config.ratio_range[0] <= instance_ratio < Config.ratio_range[1]:
                continue
            # 选得搜索图片
            instance_img = self.imread(instance_whole_path_name)
            # 进行图片随机色彩空间转换，一种数据增强
            if np.random.rand(1) < Config.gray_ratio:  # 这里为什么要转了2次
                exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_RGB2GRAY)
                exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_GRAY2RGB)
                instance_img = cv2.cvtColor(instance_img, cv2.COLOR_RGB2GRAY)
                instance_img = cv2.cvtColor(instance_img, cv2.COLOR_GRAY2RGB)
            if Config.exemplar_stretch:
                # 再次缩放尺寸
                exemplar_img, exemplar_gt_w, exemplar_gt_h = self.randomStretch(exemplar_img, exemplar_gt_w,
                                                                                exemplar_gt_h)
            # 若有放缩的操作后，要保证尺寸要回到规定的尺寸
            exemplar_img, _ = crop_and_pad(exemplar_img, (exemplar_img.shape[0] - 1) / 2,
                                           (exemplar_img.shape[1] - 1) / 2, self.center_crop_size,
                                           self.center_crop_size)
            instance_img, instance_gt_w, instance_gt_h = self.randomStretch(instance_img, instance_gt_w,
                                                                            instance_gt_h)
            img_h, img_w, _ = instance_img.shape
            cx_origin, cy_origin = (img_w - 1) / 2, (img_h - 1) / 2
            # 加中心偏移，减轻只在中心找目标的趋势
            cx_add_shift, cy_add_shift = cx_origin + np.random.randint(-self.max_shift, self.max_shift), \
                                         cy_origin + np.random.randint(-self.max_shift, self.max_shift)
            instance_img, scale = crop_and_pad(instance_img, cx_add_shift, cy_add_shift,
                                               self.random_crop_size, self.random_crop_size)
            # x和y的相对位置，相对于中心点的
            instance_gt_shift_cx, instance_gt_shift_cy = cx_origin - cx_add_shift, cy_origin - cy_add_shift
            exemplar_img, instance_img = self.z_transforms(exemplar_img), self.x_transforms(instance_img)
            # 训练时，将回归分支锚框和gt的差返回，将分类分支锚框label返回
            # 这里的cx，cy，以及训练时预测的cx，cy都是相对位置，相对于中心点的
            regression_target, cls_label_map = self.compute_target(self.anchors,
                                                                   np.array(list(map(round,
                                                                                     [instance_gt_shift_cx,
                                                                                      instance_gt_shift_cy,
                                                                                      instance_gt_w,
                                                                                      instance_gt_h]))))
            return exemplar_img, instance_img, regression_target, cls_label_map.astype(np.int64)

    def imread(self, img_dir):
        img = cv2.imread(img_dir)
        return img

    def sample_weights(self, center, low_idx, high_idx, sample_type='uniform'):
        """
         采样权重
        """
        weights = list(range(low_idx, high_idx))
        weights.remove(center)
        weights = np.array(weights)
        if sample_type == 'linear':
            weights = abs(weights - center)
        if sample_type == 'sqrt':
            weights = np.sqrt(abs(weights - center))
        if sample_type == 'uniform':
            weights = np.ones_like(weights)
        return weights / sum(weights)

    def randomStretch(self, origin_img, gt_w, gt_h):
        h, w = origin_img.shape[:2]
        origin_img = RandomStretch(origin_img)
        scaled_h, scaled_w = origin_img.shape[:2]
        scale_ratio_w, scale_ratio_h = scaled_w / w, scaled_h / h
        gt_w = scale_ratio_w * gt_w
        gt_h = scale_ratio_h * gt_h
        return origin_img, gt_w, gt_h

    def compute_target(self, anchors, box):
        """
        :param box: 这里box的cx，cy是相对于中心点的相对位置
        :return: regression_target 4位偏移量 ， cls_label_map 1位标签0或1
        """
        regression_target = box_delta_in_gt_anchor(anchors, box)
        anchors_iou = compute_iou(anchors, box)
        pos_index = np.where(anchors_iou > Config.iou_pos_threshold)[0]
        neg_index = np.where(anchors_iou < Config.iou_neg_threshold)[0]
        cls_label_map = np.ones_like(anchors_iou) * -1
        cls_label_map[pos_index] = 1
        cls_label_map[neg_index] = 0
        return regression_target, cls_label_map

    def __len__(self):
        return self.num
