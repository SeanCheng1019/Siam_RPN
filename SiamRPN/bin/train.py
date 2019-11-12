import numpy as np
import cv2
import os
from glob import glob
import torch as t
from sklearn.model_selection import train_test_split
from net.config import Config
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from lib.custom_transforms import RandomStretch, ToTensor
from lib.dataset import GetDataSet
from net.net_siamrpn import SiameseAlexNet
import pickle


def train(data_dir, model_path=None, vis_port=None, init=None):
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
    valid_z_transforms = transforms.Compose([
        ToTensor()
    ])
    valid_x_transforms = transforms.Compose([
        ToTensor()
    ])

    # get train dataset
    train_dataset = GetDataSet(train_sequences, data_dir, train_z_transforms, train_x_transforms, meta_data,
                               training=True)
    # get valid dataset
    valid_dataset = GetDataSet(valid_sequences, data_dir, valid_z_transforms, valid_x_transforms, meta_data,
                               training=False)
    # 创建dataloader迭代器
    trainloader = DataLoader(train_dataset, batch_size=Config.train_batch_size * t.cuda.device_count(),
                             shuffle=True, pin_memory=True,
                             num_workers=Config.train_num_workers * t.cuda.device_count(),
                             drop_last=True)
    validloader = DataLoader(valid_dataset, batch_size=Config.valid_batch_size * t.cuda.device_count(),
                             shuffle=False, pin_memory=True,
                             num_workers=Config.valid_num_workers * t.cuda.device_count(), drop_last=True)
    # 创建summary writer
    if not os.path.exists(Config.log_dir):
        os.mkdir(Config.log_dir)
    summary_writer = SummaryWriter(Config.log_dir)

    # start training
    model = SiameseAlexNet()
    model = model.cuda()
    optimizer = t.optim.SGD(model.parameters, lr=Config.lr, momentum=Config.momentum,
                            wehight_decay=Config.weight_dacay)
    # load model weight
    if model_path and init:
        print("init training with checkpoint %s" % model_path + '\n')
        print('--------------------------------------------------------------------------------- \n')
        # 这里load的是整个模型，包括网络、优化方法等等
        checkpoint = t.load(model_path)
        if 'model' in checkpoint.keys():
            # 这里加载的是网络的
            model.load_state_dict(checkpoint['model'])
        # 换个方式加载
        else:
            model_dict = model.state_dict()  # 啥意思
            model_dict.update(checkpoint)
            model.load_state_dict(model_dict)
        del checkpoint

