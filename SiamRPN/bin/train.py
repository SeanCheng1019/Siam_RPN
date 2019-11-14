import numpy as np
import cv2
import os
from glob import glob
import torch as t
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from net.config import Config
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from tensorboardX import SummaryWriter
from collections import OrderedDict
from lib.custom_transforms import RandomStretch, ToTensor
from lib.dataset import GetDataSet
from lib.loss import rpn_cross_entropy_banlance, rpn_smoothL1
from lib.util import ajust_learning_rate
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
    anchors = train_dataset.anchors
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
    start_epoch = 1
    # load model weight
    if model_path and init:  # 需要初始化以及存在训练模型时
        print("init training with checkpoint %s" % model_path + '\n')
        print('--------------------------------------------------------------------------------- \n')
        # 这里load的是整个模型，包括网络、优化方法等等
        checkpoint = t.load(model_path)
        if 'model' in checkpoint.keys():
            # 这里加载的是网络的
            model.load_state_dict(checkpoint['model'])
        # 换个方式加载
        else:
            model_dict = model.state_dict()  # state_dict返回的是整个网络的状态的字典
            model_dict.update(checkpoint)
            model.load_state_dict(model_dict)
        del checkpoint
        # 只有执行完下面这句，显存才会在Nvidia-smi中释放
        t.cuda.empty_cache()
        print("finish initing checkpoint! \n")
    elif model_path and not init:  # 无需初始化且有之前断点保存的模型时
        print("loading the previous checkpoint %s" % model_path + '\n')
        print('-------------------------------------------------------------------------------- \n')
        checkpoint = t.load(model_path)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint
        t.cuda().empty_cache()
        print("finish loading previous checkpoint! \n")
    elif not model_path and Config.pretrained_model:  # 需加载预训练模型的时候
        print("load pre-trained checkpoint %s" % Config.pretrained_model + '\n')
        print('-------------------------------------------------------------------------------- \n')
        checkpoint = t.load(Config.pretrained_model)
        checkpoint = {k.replace('features.features', 'sharedFeatExtra'): v for k, v in checkpoint.items()}
        model_dict = model.state_dict()
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
        print("finish loading pre-trained model \n")

    # 训练的时候前3个层的参数是固定的
    def freeze_layers(model):
        for layer in model.sharedFeatExtra[:10]:
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()  # 由于参数固定，所以这层的bn相当于是测试模式
                for k, v in layer.named_parameters():
                    v.requires_grad = False
            elif isinstance(layer, nn.Conv2d):
                for k, v in layer.named_parameters():
                    v.requires_grad = False
            elif isinstance(layer, nn.MaxPool2d):
                continue
            elif isinstance(layer, nn.ReLU):
                continue
            else:
                raise KeyError("something wrong in fixing 3 layers \n")
            print("fixed layers: " + model.sharedFeatExtra[:10])

    if t.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    for epoch in range(start_epoch, Config.epoch + 1):
        train_loss = []
        model.train()  # 设置为训练模式 train=True
        if Config.fix_former_3_layers:
            if t.cuda.device_count() > 1:  # 如果GPU数量大于1，这样是什么意思？
                freeze_layers(model.module)
            else:
                freeze_layers(model)
        # 为了训练时在终端打印实时loss设置的
        loss_temp_cls = 0
        loss_temp_reg = 0
        for i, data in enumerate(tqdm(trainloader)):
            exemplar_imgs, instance_imgs, regression_target, cls_label_map = data
            regression_target, cls_label_map = regression_target.cuda(), cls_label_map.cuda()
            pred_cls_score, pred_regression = model(exemplar_imgs.cuda(), instance_imgs.cuda())
            pred_cls_score = pred_cls_score.reshape(-1, 2,
                                                    Config.anchor_num *
                                                    Config.score_map_size *
                                                    Config.score_map_size).permute(0, 2, 1)
            pred_regression = pred_regression.reshape(-1, 4,
                                                      Config.anchor_num * Config.score_map_size *
                                                      Config.score_map_size).permute(0, 2, 1)
            cls_loss = rpn_cross_entropy_banlance(pred_cls_score, cls_label_map, Config.num_pos,
                                                  Config.num_neg, anchors,
                                                  ohem_pos=Config.ohem_pos, ohem_neg=Config.ohem_neg)
            reg_loss = rpn_smoothL1(pred_regression, regression_target, cls_label_map,
                                    Config.num_pos, ohem=Config.ohem_reg)
            loss = cls_loss + Config.lamb * reg_loss
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播求梯度
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), Config.clip)
            # 更新参数
            optimizer.step()
            step = (epoch - 1) * len(trainloader) + i
            summary_writer.add_scalar('train/cls_loss', cls_loss.data, step)
            summary_writer.add_scalar('train/reg_loss', reg_loss.data, step)
            # 加入总loss
            train_loss.append(loss.detach().cpu())
            loss_temp_cls += cls_loss.detach().cpu().numpy()
            loss_temp_reg += reg_loss.detach().cpu().numpy()
            if (i + 1) % Config.show_interval == 0:
                tqdm.write("[epoch %2d][iter %4d] cls_loss: %.4f, reg_loss: %.4f, lr: %.2e"
                           % (epoch, i, loss_temp_cls / Config.show_interval,
                              loss_temp_reg / Config.show_interval, optimizer.param_group[0]['lr']))
                loss_temp_cls = 0
                loss_temp_reg = 0
                if vis_port:
                    pass
        train_loss = np.mean(train_loss)

        # finish training an epoch, starting validation
        valid_loss = []
        model.eval()
        for i, data in enumerate(tqdm(validloader)):
            exemplar_imgs, instance_imgs, regression_target, cls_label_map = data
            regression_target, cls_label_map = regression_target.cuda(), cls_label_map.cuda()
            pred_regression, pred_cls_score = model(exemplar_imgs.cuda(), instance_imgs.cuda())
            pred_cls_score = pred_cls_score.reshape(-1, 2,
                                                    Config.anchor_num *
                                                    Config.score_map_size *
                                                    Config.score_map_size).permute(0, 2, 1)
            pred_regression = pred_regression.reshape(-1, 4,
                                                      Config.anchor_num * Config.score_map_size *
                                                      Config.score_map_size).permute(0, 2, 1)
            cls_loss = rpn_cross_entropy_banlance(pred_cls_score, cls_label_map, Config.num_pos,
                                                  Config.num_neg, anchors, ohem_pos=Config.ohem_pos,
                                                  ohem_neg=Config.ohem_neg)
            loss = pred_regression + Config.lamb * cls_loss
            valid_loss.append(loss.detach().cpu())
        valid_loss = np.mean(valid_loss)
        print("[EPOCH %2d] valid_loss: %.4f, train_loss: %.4f" % (epoch, valid_loss, train_loss))
        # 这里验证集的add_scalar的step参数和之前训练时候的不同
        summary_writer.add_scalar('valid/loss', valid_loss, (epoch + 1) * len(trainloader))
        ajust_learning_rate(optimizer, Config.gamma)
        # save model
        if epoch % Config.save_interval == 0:
            if not os.path.exists('./data/models/'):
                os.mkdir('./data/models/')
            save_name = './data/models/siamrpn_epoch_{}.pth'.format(epoch)
            if t.cuda.device_count() > 1:
                new_state_dict = OrderedDict()
                for k, v in model.state_dict().items():
                    new_state_dict[k] = v
            new_state_dict = model.state_dict()
            t.save({
                'epoch': epoch,
                'model': new_state_dict,
                'optimizer': optimizer.state_dict(),
            }, save_name)
            print('save model as:{}'.format(save_name))
