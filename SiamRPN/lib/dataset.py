from torch.utils.data.dataset import Dataset
import numpy as np
from net.config import Config
from lib.util import generate_anchors


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
        self.num = len(sequence_names) if not training else \
            Config.pairs_per_sequence_per_epoch * len(sequence_names)
        for track_sequence_name in self.meta_data.keys():
            track_sequence_info = self.meta_data[track_sequence_name]
            for object_id in track_sequence_info.keys():
                if len(track_sequence_info[object_id]) < 2:
                    del track_sequence_info[object_id]
        self.anchors = generate_anchors()

    def __getitem__(self, index):  # 何时调用的，何时传入的index参数
        all_idx = np.arange(self.num)
        np.random.shuffle(all_idx)
        for idx in all_idx:
            index = index % len(self.sequence_names)

    def __len__(self) -> int:
        return super().__len__()

