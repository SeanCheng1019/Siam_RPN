import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import random


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
    :param input: (N, 15*15*5, 2)
    :param target: (15*15*5, )
    :param num_pos:
    :param num_neg:
    :param anchors:
    :param ohem_pos:  Online Hard Example Mining positive
    :param ohem_neg:  Online Hard Example Mining negative
    :return:
    '''
    loss_all = []
    for batch_id in range(target[0]):
        pos_index = np.where(target[batch_id].cpu() == 1)[0]  # type: ndarray
        neg_index = np.where(target[batch_id].cpu() == 0)[0]
        # 正样本数不一定有num_pos规定的这么多 所以要先看实际情况的正样本有多少
        min_pos = min(len(pos_index), num_pos)
        # 当正样本数量不足num_pos规定的这么多时，负样本数也要按照正负样本比例做调整
        min_neg = int(min(len(pos_index) * num_neg / num_pos, num_neg))
        if ohem_pos:
            pass
        else:
            # 随机选出min_pos个数的正样本
            if len(pos_index) > 0:
                pos_index_random = random.sample(pos_index.tolist(), min_pos)
                pos_loss_final = F.cross_entropy(input=input[batch_id][pos_index_random],
                                                 target=target[batch_id][pos_index_random], reduction='none')
            else:
                pos_loss_final = t.FloatTensor([0]).cuda()
        if ohem_neg:
            pass
        else:
            if len(pos_index) > 0:
                neg_index_random = random.sample(neg_index.tolist(), min_neg)
                neg_loss_final = F.cross_entropy(input=input[batch_id][neg_index_random],
                                                 target=target[batch_id][neg_index_random], reduction='none')
            else:
                # 全部是负样本
                neg_index_random = random.sample(neg_index.tolist(), num_neg)
                neg_loss_final = F.cross_entropy(input=input[batch_id][neg_index_random],
                                                 target=target[batch_id][neg_index_random], reduction='none')
        loss = (pos_loss_final.mean() + neg_loss_final.mean()) / 2
        loss_all.append(loss)
    final_loss = t.stack(loss_all).mean()
    return final_loss



