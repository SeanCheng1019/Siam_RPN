import torch.nn as nn
from net.config import Config
import torch as t
from torch.autograd import Variable
from net.feature_align import FeatureAlign
import numpy as np


class STMM(nn.Module):

    def __init__(self, N, T, D, M, MULT):
        """
        :param N: seq_per_batch
        :param T: time step_per_batch
        :param D: input channel
        :param M: hidden channel (memory)
        :param MULT:
        """
        super(STMM, self).__init__()
        self.N = N
        self.T = T
        self.D = D
        self.M = M
        self.MULT = MULT or 1
        cell = STMM_cell(D, M)
        self.cell = cell

    def forward(self, x, mem=None):
        N_, C, H, W = x.shape
        N, T, M = int(N_ / Config.his_window), self.T, self.M
        feat_input = x.reshape(N, T, C, H, W)
        if mem == None:
            mem = Variable(t.zeros([N, M, H, W])).cuda()
        self.mem_input = []
        self.mem_output = []
        self.z_output = []
        self.r_output = []
        for time in range(T):
            feat_node = feat_input[:, time, :, :, :]
            if time == 0:
                prev_feat = feat_input[:, time, :, :, :]
            else:
                prev_feat = feat_input[:, time - 1, :, :, :]
            self.mem_input.append(mem)
            mem0, z, r = self.cell(feat_node, mem, prev_feat)
            mem = mem0  # update mem , 给下一次输入做准备
            self.mem_output.append(mem0)
            self.z_output.append(z)
            self.r_output.append(r)
        mem_output = [x.cpu().detach().numpy() for x in self.mem_output]
        output = t.from_numpy(np.stack(mem_output).reshape(T, N, M, H, W).transpose(1, 0, 2, 3, 4)).contiguous()
        return output


class STMM_cell(nn.Module):
    def __init__(self, D, M):
        super().__init__()
        self.conv_w = nn.Conv2d(D, M, 3, 1, 1)
        self.conv_u = nn.Conv2d(M, M, 3, 1, 1, bias=False)
        self.conv_z_w = nn.Conv2d(D, M, 3, 1, 1)
        self.conv_z_u = nn.Conv2d(M, M, 3, 1, 1, bias=False)
        self.conv_r_w = nn.Conv2d(D, M, 3, 1, 1)
        self.conv_r_u = nn.Conv2d(M, M, 3, 1, 1, bias=False)
        self.FeatureAlign = FeatureAlign(k=3)
    def forward(self, feat_input, prev_mem, prev_feat):
        """
        :param feat_input:
        :param prev_mem:
        :param prev_feat:
        :return:
        """
        mem0 = prev_mem
        if Config.memAlign:
            mem0 = self.FeatureAlign.forward(feat_input, prev_feat, prev_mem)
        else:
            # 先随便设的参数
            feat_input = feat_input + 0 * prev_feat

        #  特征对齐   (这里的relu是临时的，还需要改成和论文里一样的relu)
        z = t.relu(self.conv_z_w(feat_input) + self.conv_z_u(mem0))
        r = t.relu(self.conv_r_w(feat_input) + self.conv_r_u(mem0))
        mem_ = t.relu(self.conv_w(feat_input) + self.conv_u(r * mem0))
        mem = (1 - z) * mem0 + z * mem_

        return mem, z, r
