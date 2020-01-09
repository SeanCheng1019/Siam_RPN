import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import random
from lib.util import nms, normalization


def rpn_cross_entropy(input, target):
    '''
    :param input: (15*15*5, 2)
    :param target: (15*15*5, )
    :return:
    '''
    mask_ignore = target == -1
    mask_select = (1 - mask_ignore).astype(bool)
    loss = F.cross_entropy(input=input[mask_select], target=target[mask_select])
    return loss


def rpn_cross_entropy_banlance(input, target, num_pos, num_neg, anchors, ohem_pos=None, ohem_neg=None):
    '''
    :param input: (N, 19*19*5, 2)
    :param target: (N, 19*19*5, )
    :param num_pos:
    :param num_neg:
    :param anchors:
    :param ohem_pos:  Online Hard Example Mining positive
    :param ohem_neg:  Online Hard Example Mining negative
    :return:
    '''
    loss_all = []
    for batch_id in range(target.shape[0]):
        pos_index = np.where(target[batch_id].cpu() == 1)[0]  # type: ndarray
        neg_index = np.where(target[batch_id].cpu() == 0)[0]
        # 正样本数不一定有num_pos规定的这么多 所以要先看实际情况的正样本有多少
        min_pos = min(len(pos_index), num_pos)
        # 当正样本数量不足num_pos规定的这么多时，负样本数也要按照正负样本比例做调整
        min_neg = int(min(len(pos_index) * num_neg / num_pos, num_neg))
        if ohem_pos:
            if len(pos_index) > 0:
                if len(pos_index) == 1:
                    pos_loss_temp = F.cross_entropy(input=input[batch_id][pos_index.tolist()],
                                                    target=target[batch_id][pos_index.tolist()[0]],
                                                    reduction='none')
                else:
                    # 先对所有都做交叉熵损失
                    pos_loss_temp = F.cross_entropy(input=input[batch_id][pos_index.tolist()],
                                                    target=target[batch_id][pos_index.tolist()].squeeze(),
                                                    reduction='none')
                # 用nms去除非极大值的框
                selected_pos_index = nms(anchors[pos_index.tolist()],
                                         pos_loss_temp.cpu().detach().numpy(), min_pos)
                pos_loss_final = pos_loss_temp[selected_pos_index]
            else:
                pos_loss_final = t.FloatTensor([0]).cuda()
        else:
            # 随机选出min_pos个数的正样本
            if len(pos_index) > 0:
                pos_index_random = random.sample(pos_index.tolist(), min_pos)
                # print("处理loss \n", pos_index_random, input[batch_id][pos_index_random].shape,
                #       target[batch_id][pos_index_random].squeeze().shape, target[batch_id][pos_index_random].squeeze(), "\n")
                # 为了处理cross_entropy的维度问题,出现在正样本个数只有1个情况下
                if len(pos_index_random) == 1:
                    # print("处理1个的情况")
                    print(pos_index_random, "\n")
                    print(batch_id, "\n")
                    print(target.shape, "\n")
                    print(pos_index_random[0])
                    print("size: ", input[batch_id][pos_index_random].shape,
                          target[batch_id][pos_index_random[0]].shape)

                    print(target[batch_id][pos_index_random[0]].unsqueeze(0).shape)
                    pos_loss_final = F.cross_entropy(input=input[batch_id][pos_index_random],
                                                     target=target[batch_id][pos_index_random[0]].unsqueeze(0), reduction='none')
                else:
                    # 这里的交叉熵损失 先对每一个anchor的两个分数做softmax，即两个分数加起来和为1
                    # 这里因为规定正样本标为1，负样本标为0，而“1”正好是第2列，所以朝着第二列是正样本分数的方向训练，所以第二列是看做正样本的分数。
                    pos_loss_final = F.cross_entropy(input=input[batch_id][pos_index_random],
                                                     target=target[batch_id][pos_index_random].squeeze(),
                                                     reduction='none')
            else:
                pos_loss_final = t.FloatTensor([0]).cuda()
        if ohem_neg:
            if len(pos_index) > 0:
                neg_loss_temp = F.cross_entropy(input=input[batch_id][neg_index.tolist()],
                                                target=target[batch_id][neg_index.tolist()].squeeze(), reduction='none')
                selected_neg_index = nms(anchors[neg_index], neg_loss_temp.cpu().detach().numpy(), min_neg)
                neg_loss_final = neg_loss_temp[selected_neg_index]
            else:
                # 只有负样本
                neg_loss_temp = F.cross_entropy(input=input[batch_id][neg_index.tolist()],
                                                target=target[batch_id][neg_index.tolist()].squeeze(), reduction='none')
                selected_neg_index = nms(anchors[neg_index], neg_loss_temp.cpu().detach().numpy(), num_neg)
                neg_loss_final = neg_loss_temp[selected_neg_index]
        else:
            # 随机选出min_neg个数的负样本
            if len(pos_index) > 0:
                neg_index_random = random.sample(neg_index.tolist(), min_neg)
                neg_loss_final = F.cross_entropy(input=input[batch_id][neg_index_random],
                                                 target=target[batch_id][neg_index_random].squeeze(), reduction='none')
            # 对所有的负样本进行损失计算
            else:
                # 全部是负样本
                neg_index_random = random.sample(neg_index.tolist(), num_neg)
                neg_loss_final = F.cross_entropy(input=input[batch_id][neg_index_random],
                                                 target=target[batch_id][neg_index_random].squeeze(), reduction='none')
        # # 加loss归一化
        # pos_loss_final = normalization(pos_loss_final)
        # neg_loss_final = normalization(neg_loss_final)
        loss = (pos_loss_final.mean() + neg_loss_final.mean()) / 2
        loss_all.append(loss)
    final_loss = t.stack(loss_all).mean()
    return final_loss


def rpn_smoothL1(input, target, label, num_pos, ohem=None):
    """
    :param input: torch.size([1,15*15*5,4])
    :param target: torch.size([1,15*15*5,4])
    :param label: torch.size([1,15*15*5])
    """
    loss_all = []
    for batch_id in range(target.shape[0]):
        # 只对正样本进行回归
        pos_index = np.where(label[batch_id].cpu() == 1)[0]
        min_pos = min(len(pos_index), num_pos)
        if ohem:
            if len(pos_index) > 0:
                # loss的维度是？
                loss = F.smooth_l1_loss(input=input[batch_id][pos_index],
                                        target=target[batch_id][pos_index], reduction='none')
                sort_index = t.argsort(loss.mean(1))
                loss_ohem = loss[sort_index[-num_pos:]]
            else:
                loss_ohem = t.FloatTensor([0]).cuda()[0]
            loss_all.append(loss_ohem.mean())

        else:
            pos_index = random.sample(pos_index.tolist(), min_pos)
            if len(pos_index) > 0:
                loss = F.smooth_l1_loss(input=input[batch_id][pos_index],
                                        target=target[batch_id][pos_index])
            else:
                loss = t.FloatTensor([0]).cuda()[0]
            loss_all.append(loss.mean())
    final_loss = t.stack(loss_all).mean()
    return final_loss
