import numpy as np
import cv2
import os
from glob import glob
from sklearn.model_selection import train_test_split
from net.config import Config
from torchvision.transforms import transforms
from lib.custom_transforms import RandomStretch, ToTensor
from lib.dataset import GetDataSet
import pickle


def train(data_dir, model_path=None, vis_flag=None, init=None):
    # 得到所有视频序列（已处理）
    meta_data_path = os.path.join(data_dir, "meta_data.pkl")
    meta_data = pickle.load(open(meta_data_path, 'rb'))
    all_sequences = [x[0] for x in meta_data]
    # 分割出训练集、测试集
    train_sequences, valid_sequences = train_test_split(all_sequences,
                                                        test_size=1 - Config.train_ratio, random_state=Config.seed)
    # define transforms
    train_z_transforms = transforms.Compose([
        ToTensor()
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

    # get train dataset
    train_dataset = GetDataSet(train_sequences, data_dir, train_z_transforms, train_x_transforms, meta_data,
                               training=True)

    # get valid dataset
