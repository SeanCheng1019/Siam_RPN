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
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from collections import OrderedDict
from lib.custom_transforms import RandomStretch, ToTensor
from lib.dataset import GetDataSet
from lib.loss import rpn_cross_entropy_banlance, rpn_smoothL1
from lib.util import ajust_learning_rate, get_topK_box, add_box_img, compute_iou, box_transform_use_reg_offset,crop_and_pad
from net.net_siamrpn import SiameseAlexNet
from lib.viusal import visual
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
    train_batch_size = Config.stmm_train_batch_size if Config.update_template else Config.train_batch_size
    valid_batch_size = Config.stmm_valid_batch_size if Config.update_template else Config.valid_batch_size
    trainloader = DataLoader(train_dataset, batch_size=train_batch_size * t.cuda.device_count(),
                             shuffle=True, pin_memory=True,
                             num_workers=Config.train_num_workers * t.cuda.device_count(),
                             drop_last=True)
    validloader = DataLoader(valid_dataset, batch_size=valid_batch_size * t.cuda.device_count(),
                             shuffle=False, pin_memory=True,
                             num_workers=Config.valid_num_workers * t.cuda.device_count(), drop_last=True)
    # 创建summary writer
    if not os.path.exists(Config.log_dir):
        os.mkdir(Config.log_dir)
    summary_writer = SummaryWriter(Config.log_dir)
    # 可视化
    if vis_port:
        vis = visual()
    # start training
    model = SiameseAlexNet()
    model = model.cuda()
    optimizer = t.optim.SGD(model.parameters(), lr=Config.lr, momentum=Config.momentum,
                            weight_decay=Config.weight_dacay)
    start_epoch = 1
    # load model weight
    if model_path and init:  # 需要初始化以及存在训练模型时
        print("init training with checkpoint %s" % model_path + '\n')
        print('--------------------------------------------------------------------------------- \n')
        # 这里load的是整个模型，包括网络、优化方法等等
        checkpoint = t.load(model_path)
        if 'model' in checkpoint.keys():
            # 这里加载的是网络的pred_cls_score
            # 这里加载的是网络的pred_cls_score
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
        t.cuda.empty_cache()
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
            print("fixed layers:  \n", layer)

    if t.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    for epoch in range(start_epoch, Config.epoch + 1):
        print("staring epoch{} \n".format(epoch))
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
        loss_temp_template = 0
        for i, data in enumerate(tqdm(trainloader)):
            # 每次加载一个mini-batch数量的样本
            # exemplar_imgs size:[32,127,127,3]  regression_target size:[32,1805,4]
            exemplar_imgs, instance_imgs, regression_target, cls_label_map, instance_his_imgs = data
            if Config.update_template:  # 这里收集的历史搜索帧的大小已经裁剪成 模板大小
                instance_his_imgs = [x.numpy() for x in instance_his_imgs]
                instance_his_imgs = np.stack(instance_his_imgs).transpose(1, 0, 2, 3, 4)
                instance_his_imgs = instance_his_imgs.reshape(-1, Config.exemplar_size, Config.exemplar_size, 3)
                # exemplar_imgs = np.concatenate(exemplar_imgs, axis=0)  # 合并第一第二维度，因为网络的输入规定四维
                instance_his_imgs = t.from_numpy(instance_his_imgs)
                pred_cls_score, pred_regression, template_loss = model(exemplar_imgs.cuda(),
                                                        instance_imgs.cuda(),
                                                        instance_his_imgs,
                                                        training=True)
            else:
                pred_cls_score, pred_regression = model(exemplar_imgs.cuda(),
                                                        instance_imgs.cuda(),
                                                        instance_his_imgs,
                                                        training=True)
            regression_target, cls_label_map = regression_target.cuda(), cls_label_map.cuda()


            pred_cls_score = pred_cls_score.reshape(-1, 2,
                                                    Config.anchor_num *
                                                    Config.train_map_size *
                                                    Config.train_map_size).permute(0, 2, 1)

            pred_regression = pred_regression.reshape(-1, 4,
                                                      Config.anchor_num * Config.train_map_size *
                                                      Config.train_map_size).permute(0, 2, 1)

            cls_loss = rpn_cross_entropy_banlance(pred_cls_score, cls_label_map, Config.num_pos,
                                                  Config.num_neg, anchors,
                                                  ohem_pos=Config.ohem_pos, ohem_neg=Config.ohem_neg)
            reg_loss = rpn_smoothL1(pred_regression, regression_target, cls_label_map,
                                    Config.num_pos, ohem=Config.ohem_reg)
            # 总的loss上加上模版loss
            if Config.update_template:
                loss = cls_loss + Config.lamb * reg_loss + template_loss
            else:
                loss = cls_loss + Config.lamb * reg_loss
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播求梯度
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), Config.clip)
            # 更新参数
            optimizer.step()
            step = (epoch - 1) * len(trainloader) + i
            # summary_writer.add_scalar('train/cls_loss', cls_loss.data, step)
            # summary_writer.add_scalar('train/reg_loss', reg_loss.data, step)
            if Config.update_template:
                summary_writer.add_scalars('train',
                                           {'cls_loss': cls_loss.data.item(), 'reg_loss': reg_loss.data.item(),
                                            'template_loss': template_loss.data.item(),
                                            'total_loss': loss.data.item()},
                                           step)
            else:
                summary_writer.add_scalars('train',
                                           {'cls_loss': cls_loss.data.item(), 'reg_loss': reg_loss.data.item(),
                                            'total_loss': loss.data.item()},
                                           step)
            # 加入总loss
            train_loss.append(loss.detach().cpu())
            loss_temp_cls += cls_loss.detach().cpu().numpy()
            loss_temp_reg += reg_loss.detach().cpu().numpy()
            loss_temp_template += template_loss.detach().cpu().numpy()
            if (i + 1) % Config.show_interval == 0:
                tqdm.write("[epoch %2d][iter %4d] cls_loss: %.4f, reg_loss: %.4f, temp_loss: %.4f, lr: %.2e"
                           % (epoch, i, loss_temp_cls / Config.show_interval,
                              loss_temp_reg / Config.show_interval, loss_temp_template / Config.show_interval,
                              optimizer.param_groups[0]['lr']))
                loss_temp_cls = 0
                loss_temp_reg = 0
                loss_temp_template = 0

                # 可视化
                if vis_port:
                    anchors_show = train_dataset.anchors
                    exem_img = exemplar_imgs[0].cpu().detach().numpy()
                    inst_img = instance_imgs[0].cpu().detach().numpy()
                    # choose odd layer and show the heatmap
                    # cls_response = cls_map_vis.squeeze()[0:10, :, :]
                    # cls_res_show = []
                    # for x in range(10):
                    #     if x % 2 == 1:
                    #         res = cls_response[x:x + 1, :, :].squeeze().cpu().detach().numpy()
                    #         cls_res_show.append(res)
                    # count = 20
                    # for heatmap in cls_res_show:
                    #     vis.plot_heatmap(heatmap, win=count)
                    #     count += count
                    topk = Config.show_topK
                    vis.plot_img(exem_img.transpose(2, 0, 1), win=1, name='exemplar_img')
                    cls_pred = cls_label_map[0]  # 对这个存疑,看看cls_pred的内容
                    gt_box = get_topK_box(cls_pred, regression_target[0], anchors_show)[0]
                    # show gt box
                    img_box = add_box_img(inst_img, gt_box, color=(255, 0, 0))
                    vis.plot_img(img_box.transpose(2, 0, 1), win=2, name='instance_img')
                    # show anchor with max score (without regression)
                    cls_pred = F.softmax(pred_cls_score, dim=2)[0, :, 1]  # 1 的意思是最后一维，第一个代表的是正样本结果
                    scores, index = t.topk(cls_pred, k=topk)
                    img_box = add_box_img(inst_img, anchors_show[index.cpu()])
                    img_box = add_box_img(img_box, gt_box, color=(255, 0, 0))
                    vis.plot_img(img_box.transpose(2, 0, 1), win=3, name='max_score_anchors')

                    # max score anchor with regression
                    cls_pred = F.softmax(pred_cls_score, dim=2)[0, :, 1]
                    topk_box = get_topK_box(cls_pred, pred_regression[0], anchors_show, topk=topk)
                    img_box = add_box_img(inst_img, topk_box.squeeze())
                    img_box = add_box_img(img_box, gt_box, color=(255, 0, 0))
                    vis.plot_img(img_box.transpose(2, 0, 1), win=4, name='max_score_box')

                    # show anchor with max iou (without regression)
                    iou = compute_iou(anchors_show, gt_box).flatten()
                    index = np.argsort(iou)[-topk:]
                    img_box = add_box_img(inst_img, anchors_show[index])
                    img_box = add_box_img(img_box, gt_box, color=(255, 0, 0))
                    vis.plot_img(img_box.transpose(2, 0, 1), win=5, name='max_iou_anchor')
                    # show regressed anchor with max iou
                    reg_offset = pred_regression[0].cpu().detach().numpy()
                    topk_offset = reg_offset[index, :]
                    anchors_det = anchors_show[index, :]
                    pred_box = box_transform_use_reg_offset(anchors_det, topk_offset)
                    img_box = add_box_img(inst_img, pred_box.squeeze())
                    img_box = add_box_img(img_box, gt_box, color=(255, 0, 0))
                    vis.plot_img(img_box.transpose(2, 0, 1), win=6, name='max_iou_box')

        train_loss = np.mean(train_loss)

        # finish training an epoch, starting validation
        valid_loss = []
        model.eval()
        for i, data in enumerate(tqdm(validloader)):
            exemplar_imgs, instance_imgs, regression_target, cls_label_map, instance_his_imgs = data
            if Config.update_template:  # 这里收集的历史搜索帧的大小已经裁剪成 模板大小
                instance_his_imgs = [x.numpy() for x in instance_his_imgs]
                instance_his_imgs = np.stack(instance_his_imgs).transpose(1, 0, 2, 3, 4)
                instance_his_imgs = instance_his_imgs.reshape(-1, Config.exemplar_size, Config.exemplar_size, 3)
                # exemplar_imgs = np.concatenate(exemplar_imgs, axis=0)  # 合并第一第二维度，因为网络的输入规定四维
                instance_his_imgs = t.from_numpy(instance_his_imgs)
                pred_cls_score, pred_regression, template_loss = model(exemplar_imgs.cuda(),
                                                                       instance_imgs.cuda(),
                                                                       instance_his_imgs,
                                                                       training=False)
            else:
                pred_cls_score, pred_regression = model(exemplar_imgs.cuda(),
                                                        instance_imgs.cuda(),
                                                        instance_his_imgs)
            regression_target, cls_label_map = regression_target.cuda(), cls_label_map.cuda()
            pred_cls_score = pred_cls_score.reshape(-1, 2,
                                                    Config.anchor_num *
                                                    Config.train_map_size *
                                                    Config.train_map_size).permute(0, 2, 1)
            pred_regression = pred_regression.reshape(-1, 4,
                                                      Config.anchor_num * Config.train_map_size *
                                                      Config.train_map_size).permute(0, 2, 1)

            cls_loss = rpn_cross_entropy_banlance(pred_cls_score, cls_label_map, Config.num_pos,
                                                  Config.num_neg, anchors, ohem_pos=Config.ohem_pos,
                                                  ohem_neg=Config.ohem_neg)
            reg_loss = rpn_smoothL1(pred_regression, regression_target, cls_label_map,
                                    Config.num_pos, Config.ohem_reg)
            if Config.update_template:
                loss = cls_loss + Config.lamb * reg_loss + template_loss
            else:
                loss = cls_loss + Config.lamb * reg_loss
            valid_loss.append(loss.detach().cpu())
        valid_loss = np.mean(valid_loss)
        print("[EPOCH %2d] valid_loss: %.4f, train_loss: %.4f", (epoch, valid_loss, train_loss))
        # 这里验证集的add_scalar的step参数和之前训练时候的不同
        if Config.update_template:
            summary_writer.add_scalars('valid', {'cls_loss': cls_loss.data.item(),
                                                 'reg_loss': reg_loss.data.item(),
                                                 'template_loss': template_loss.data.item(),
                                                 'total_loss': loss.data.item()},
                                       (epoch + 1) * len(trainloader))
        else:
            summary_writer.add_scalars('valid', {'cls_loss': cls_loss.data.item(),
                                                 'reg_loss': reg_loss.data.item(),
                                                 'total_loss': loss.data.item()},
                                       (epoch + 1) * len(trainloader))
        ajust_learning_rate(optimizer, Config.gamma)
        if epoch % 10 == 0:  # 每10个epoch看一下已经选择过的序列
            print(train_dataset.choosed_idx.sort())
        # save model
        if epoch % Config.save_interval == 0:
            if not os.path.exists('../data/models/'):
                os.mkdir('../data/models/')
            if Config.update_template:
                save_name = '../data/models/siamrpn_stmm_epoch_{}.pth'.format(epoch)
            else:
                save_name = '../data/models/siamrpn_epoch_{}.pth'.format(epoch)
            if t.cuda.device_count() > 1:  # remove 'module.'
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


if __name__ == '__main__':
    # data_dir = "/home/csy/dataset/dataset/ILSVRC2015_VID_curation2"
    data_dir = "/home/csy/dataset/dataset/ILSVRC2015_VID_CURATION"
    model_path = None
    vis_port = 8097
    init = None
    train(data_dir, model_path, vis_port, init)

    """
    /media/csy/62ac73e0-814c-4dba-b59d-676690aca14b/PycharmProjects/Siam_RPN/SiamRPN/lib/loss.py:125: UserWarning: Using a target size (torch.Size([4])) that is different to the input size (torch.Size([1, 4])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
  target=target[batch_id][pos_index].squeeze())
    """
