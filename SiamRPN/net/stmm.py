import torch.nn as nn
from net.config import Config
import torch as t

class STMM(nn.Module):

    def __init__(self, N, T, D, M, MULT):
        """
        :param N: seq_per_batch
        :param T: time step_per_batch
        :param D: input data
        :param M: memory
        :param MULT:
        """
        super(STMM, self).__init__()
        self.N = N
        self.T = T
        self.D = D
        self.M = M
        self.MULT = MULT or 1
        self.cell_constructed_flag = False
        self.conv_w = nn.Conv2d(D, M, 3, 1, 1)
        self.conv_u = nn.Conv2d(M, M, 3, 1, 1, bias=False)
        self.conv_z_w = nn.Conv2d(D, M, 3, 1, 1)
        self.conv_z_u = nn.Conv2d(M, M, 3, 1, 1, bias=False)
        self.conv_r_w = nn.Conv2d(D, M, 3, 1, 1)
        self.conv_r_u = nn.Conv2d(M, M, 3, 1, 1, bias=False)

    def forward(self, feat_input, prev_feat_input=None, mem_input=None):




class STMM_cell(nn.Module):
    def __init__(self, conv_w, conv_u, conv_z_w, conv_z_u, conv_r_w,
                 conv_r_u, feat_input, prev_feat_input, mem_input):
        super().__init__()
        #  特征对齐
        if Config.memAlign:
            pass
        else:
            pass
        self.conv_w = conv_w
        self.conv_u = conv_u
        self.conv_z_u = conv_z_u
        self.conv_z_w = conv_z_w
        self.conv_r_w = conv_r_w
        self.conv_r_u = conv_r_u
    def forward(self, feat, mem):
        if mem is None:
            h = t.sigmoid(self.conv_z_w(feat)) * t.tanh(self.conv_z_u(feat))
            return h, h
        else:
            z = t.sigmoid(self.conv_z_w(feat) + self.conv_z_u(mem))
            r = t.sigmoid(self.conv_r_w(feat) + self.conv_r_u(mem))
            h_ = t.tanh(self.conv_w(feat) + self.conv_u(r * mem))
            h = (1 - z) * mem + z * h_
            return h, h


