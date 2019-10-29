import numpy as np
import cv2
import os
from glob import glob
from sklearn.model_selection import train_test_split
from net.config import Config
from torchvision.transforms import transforms
from lib.custom_transforms import RandomStretch, ToTensor


def train(data_dir, model_path=None, vis_flag=None, init=None):
    # 得到所有视频序列（已处理）
    all_sequences = glob(os.path.join(data_dir, '*'))
    # 分割出训练集、测试集
    train_sequences, valid_sequences = train_test_split(all_sequences,
                                                        test_size=1 - Config.train_ratio, random_state=Config.seed)
    # define transforms
    train_z_transforms = transforms.Compose([
        ToTensor(),
        RandomStretch()
    ])
    train_x_transforms = transforms.Compose([
        ToTensor()
    ])
    valid_z_transform = transforms.Compose([
        ToTensor()
    ])
    valid_x_transform = transforms.Compose([
        ToTensor()
    ])

    # get train data
