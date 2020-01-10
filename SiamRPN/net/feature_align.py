import torch.nn as nn
import torch.nn.functional as F
import torch as t
import numpy as np


class FeatureAlign(nn.Module):
    def __init__(self, k):
        super(FeatureAlign, self).__init__()
        assert k % 2 == 1, "k must be a odd number"
        self.k = k
        pad = (self.k - 1) / 2
        self.paddor = nn.ZeroPad2d(padding=pad)
        self.output = None
        self.masked_cpa = None

    def forward(self, cur_feat, prev_feat, prev_mem):
        # N是batch size
        N, D, H, W = prev_mem.shape
        assert prev_feat.shape[2] == H and prev_feat.shape[3] == W, \
            "Note prev_feat should have same spatial size as cur_feat"
        assert prev_mem.shape[2] == H and prev_mem.shape[3] == W, \
            "Note prev_mem should have same spatial size as cur_feat"
        pad = int((self.k - 1) / 2)
        cur_prev_affinity = []
        for idx in range(N):
            cur_feat_flat = cur_feat[idx, :, :, :].view(-1, H * W)
            prev_feat_flat = prev_feat[idx, :, :, :].view(-1, H * W)
            cur_feat_flat_norm = t.norm(cur_feat_flat, p=2, dim=0).view(H * W, 1) + 1e-8
            prev_feat_flat_norm = t.norm(prev_feat_flat, p=2, dim=0).view(1, H * W) + 1e-8
            mul_mat = t.mm(cur_feat_flat.T, prev_feat_flat)
            mul_mat = t.div(mul_mat, cur_feat_flat_norm)
            mul_mat = t.div(mul_mat, prev_feat_flat_norm)
            mul_mat_flat = mul_mat.view(-1)
            invalid_idx = np.where(mul_mat_flat.cpu().detach().numpy() < 0)[0]
            if invalid_idx.__len__() > 0:
                invalid_idx = invalid_idx.reshape(-1)
                mul_mat_flat[invalid_idx] = 0
            # record affinity
            cur_prev_affinity.append(mul_mat.view(H, W, H, W))
        self.output = \
            self.output if self.output is not None else t.zeros_like(prev_mem).view(N, D, H, W)
        # self.masked_cpa = self.masked_cpa or t.zeros_like(prev_mem).view(N, H, W, H, W)  # masked_cur_prev_affinity
        self.masked_cpa = \
            self.masked_cpa if self.masked_cpa is not None else t.zeros_like(
                t.Tensor(N, H, W, H, W))  # masked_cur_prev_affinity
        for idx in range(N):
            self.assemble(H, W, D, pad, cur_prev_affinity[idx],
                          prev_mem[idx].view(D, -1), self.output[idx].view(D, -1), self.masked_cpa[idx])
        return self.output

    def assemble(self, H, W, D, pad, cur_prev_aff, prev_mem, output, masked_cpa):
        n = D * H * W
        for index in range(n):
            HW = H * W
            d = index // HW  # 通道数号
            loc = index % HW
            y = loc // W
            x = loc % W
            bound = 10
            mass = 0.
            for i in range(-pad, pad + 1):
                for j in range(-pad, pad + 1):
                    # 点(x,y)在prev_feat_map上的相邻点
                    prev_x = x + i
                    prev_y = y + j
                    # 　当是有效的相邻点时
                    if prev_y >= 0 and prev_y < H and prev_x >= 0 and prev_x < W:
                        flat_idx = int(y * W * HW + x * HW + prev_y * W + prev_x)
                        coef = cur_prev_aff.view(-1)[flat_idx]  # 分母求和式子中的其中一个
                        if coef.cpu().detach().numpy() > 0:
                            mass += coef   # 求gamma_x,y(i,j)公式的分母

            val = 0.
            if mass > -bound and mass < bound:
                flat_idx = y * W * HW + x * HW + y * W + x
                feat_flat_idx = d * HW + y * W + x
                val = prev_mem.view(-1)[feat_flat_idx]  # 未对齐的Mt-1
                if d == 0:
                    # masked_cpa[flat_idx] += 1.0
                    pass
                else:
                    for i in (-pad, pad + 1):
                        for j in (-pad, pad + 1):
                            prev_y = y + i
                            prev_x = x + j
                            if prev_y >= 0 and prev_y < H and prev_x >= 0 and prev_x < W:
                                # Update output
                                flat_idx = y * W * HW + x * HW + prev_y * W + prev_x
                                a = cur_prev_aff.view(-1)[flat_idx]   # 求gamma_x,y(i,j)公式的分子
                                if a > 0:
                                    a = a / mass
                                    feat_flat_idx = d * HW + prev_y * W + prev_x
                                    fc = prev_mem.view(-1)[feat_flat_idx]  # Mt-1(x+i,y+j)
                                    val += a * fc  # 求对齐的Mt-1
                                    # # Update gradient
                                    # if d == 0:  # The thread for the first dim is responsible for this
                                    #     masked_cpa[flat_idx] += a
                # Get the right cell in the output
                output_idx = d * HW + y * W + x
                output.view(-1)[output_idx] = val
