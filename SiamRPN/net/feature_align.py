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

    def forward(self, cur_feat, prev_feat, prev_mem):
        N, D, H, W = prev_mem.shape
        assert prev_feat.shape[2] == H and prev_feat.shape[3] == W, \
            "Note prev_feat should have same spatial size as cur_feat"
        assert prev_mem.shape[2] == H and prev_mem.shape[3] == W, \
            "Note prev_mem should have same spatial size as cur_feat"
        pad = (self.k - 1) / 2
        cur_prev_affinity = []
        for idx in range(N):
            cur_feat_flat = cur_feat[idx, :, :, :].view(-1, H * W)
            prev_feat_flat = prev_feat[idx, :, :, :].view(-1, H * W)
            cur_feat_flat_norm = t.norm(cur_feat_flat, 0) + 1e-8
            prev_feat_flat_norm = t.norm(prev_feat_flat, 0) + 1e-8
            mul_mat = t.mm(cur_feat_flat.T, prev_feat_flat)
            mul_mat = t.div(mul_mat, cur_feat_flat_norm)
            mul_mat = t.div(mul_mat, prev_feat_flat_norm)
            mul_mat_flat = mul_mat.view(-1)
            invalid_idx = np.where(mul_mat_flat.numpy() < 0)
            if invalid_idx.__len__() > 0:
                invalid_idx = invalid_idx.reshape(-1)
                mul_mat_flat[invalid_idx] = 0
            # record affinity
            cur_prev_affinity.append(mul_mat.view(H, W, H, W))
        self.output = self.output or t.zeros_like(prev_mem).view(N, D, H, W)
        self.masked_cpa = self.masked_cpa or t.zeros_like(prev_mem).view(N, H, W, H, W)
        for idx in range(N):
            self.assemble(H, W, D, pad, cur_prev_affinity, prev_mem, self.output, self.masked_cpa)
        return self.output

    def assemble(self, H, W, D, pad, cur_prev_aff, prev_mem, output, masked_cpa):
        n = D * H * W
        for index in range(n):
            HW = H * W
            d = index / HW
            loc = index % HW
            y = loc / W
            x = loc % W
            bound = 1e-7
            mass = 0.
            for i in range(-pad, pad + 1):
                for j in range(-pad, pad + 1):
                    prev_y = y + j
                    prev_x = x + i
                    if prev_y >= 0 and prev_y < H and prev_x >= 0 and prev_x < W:
                        flat_idx = y * W * HW + x * HW + prev_y * W + prev_x
                        coef = cur_prev_aff[flat_idx]
                        if coef > 0:
                            mass += coef
            val = 0.
            if mass > -bound and mass < bound:
                flat_idx = y * W * HW + x * HW + y * W + x
                feat_flat_idx = d * HW + y * W + x
                val = prev_mem[feat_flat_idx]
                if d == 0:
                    masked_cpa[flat_idx] += 1.0
                else:
                    for i in (-pad, pad + 1):
                        for j in (-pad, pad + 1):
                            prev_y = y + i
                            prev_x = x + j
                            if prev_y >= 0 and prev_y < H and prev_x >= 0 and prev_x < W:
                                # Update output
                                flat_idx = y * W * HW + x * HW + prev_y * W + prev_x
                                a = cur_prev_aff[flat_idx]
                                if a > 0:
                                    a = a / mass
                                    feat_flat_idx = d * HW + prev_y * W + prev_x
                                    fc = prev_mem[feat_flat_idx]
                                    val += a * fc
                                    # Update gradient
                                    if d == 0:  # The thread for the first dim is responsible for this
                                        masked_cpa[flat_idx] += a
                # Get the right cell in the output
                output_idx = d * HW + y * W + x
                output[output_idx] = val

