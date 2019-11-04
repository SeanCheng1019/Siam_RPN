from torch.utils.data.dataset import Dataset
import numpy as np
from net.config import Config
from lib.util import generate_anchors
from glob import glob
import os


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
            for object_id in track_sequence_info.keys():
                if len(track_sequence_info[object_id]) < 2:
                    del track_sequence_info[object_id]
        self.training = training
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
            exemplar_name = glob(os.path.join(self.data_dir, sequence, trk_frames[exemplar_index] +
                                              ".{:02d}.patch*.jpg".format(trkid)))[0]
            exemplar_gt_w, exemplar_gt_h, exemplar_img_w, exemplar_img_h = \
                float(exemplar_name.split('/')[-1].split('_')[2]), float(exemplar_name.split('/')[-1].split('_')[4][1:])
            float(exemplar_name.split('/')[-1].split('_')[7]), float(exemplar_name.split('/')[-1].split('_')[-1][:-4])


def __len__(self) -> int:
    return super().__len__()
