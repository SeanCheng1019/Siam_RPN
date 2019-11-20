import os
import cv2
import numpy as np
from glob import glob


class Video(object):
    """
    处理跟踪单个视频序列
    """

    def __init__(self, name, root, video_dir, init_rect,
                 img_names, gt_rect, attr, load_img=False):
        """
        :param init_rect:  第一帧target box
        :param img_names: 帧名
        :param gt_rect: gt
        :param attr:
        :param load_img: 是否要加载全部图片的flag
        """
        self.name = name
        self.video_dir = video_dir
        self.init_rect = init_rect
        self.gt_traj = gt_rect
        self.attr = attr
        self.pred_trajs = {}
        self.img_names = [os.path.join(root, x) for x in img_names]
        self.imgs = None

        if load_img:
            self.imgs = [cv2.imread(x) for x in self.img_names]
            self.width = self.imgs[0].shape[1]
            self.height = self.img[0].shape[0]
        else:
            img = cv2.imread(self.img_names[0])
            assert img is not None, self.img_names[0]
            self.width = img.shape[1]
            self.height = img.shape[0]

    def load_tracker(self, path, tracker_name=None, store=True):
        pass

    def load_img(self):
        if self.imgs is None:
            self.imgs = [cv2.imread(x) for x in self.img_names]
            self.width = self.imgs[0].shape[1]
            self.height = self.img[0].shape[0]

    def free_img(self):
        self.imgs = None

    def draw_box(self):
        pass

    def show(self):
        pass

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if self.imgs is None:
            return cv2.imread(self.img_names[idx]), self.gt_traj[idx]
        else:
            return self.imgs[idx], self.gt_traj[idx]

    def __iter__(self):
        for i in range(len(self.img_names)):
            if self.imgs is not None:
                yield self.imgs[i], self.gt_traj[i]
            else:
                yield cv2.imread(self.img_names[i]), self.gt_traj[i]
