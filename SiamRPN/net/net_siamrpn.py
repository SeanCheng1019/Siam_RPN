import numpy as np
from torch import nn
from net.config import Config
import torch.nn.functional as F
from net.config import Config
from lib.viusal import visual
from lib.util import norm_to_255

class SiameseAlexNet(nn.Module):
    def __init__(self, ):
        super(SiameseAlexNet, self).__init__()
        self.sharedFeatExtra = nn.Sequential(
            #conv1
            nn.Conv2d(3, 96, 11, stride=2),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(inplace=True),
            #conv2
            nn.Conv2d(96, 256, 5),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(inplace=True),
            #conv3
            nn.Conv2d(256, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            #conv4
            nn.Conv2d(384, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            #conv5
            nn.Conv2d(384, 256, 3),
            nn.BatchNorm2d(256),
        )

        self.anchor_num_per_position = Config.anchor_num
        self.input_size = Config.instance_size
        self.conv_cls1 = nn.Conv2d(256, 256 * 2 * self.anchor_num_per_position, 3)
        self.conv_cls2 = nn.Conv2d(256, 256, 3)
        self.conv_reg1 = nn.Conv2d(256, 256 * 4 * self.anchor_num_per_position, 3)
        self.conv_reg2 = nn.Conv2d(256, 256, 3)

    def forward(self, template, detection):
        N = template.size(0)
        template_feature = self.sharedFeatExtra(template)
        detection_feature = self.sharedFeatExtra(detection)
        # if Config.show_net_feature:
        #     vis = visual()
        #     detection_feature_ = norm_to_255(detection_feature[0].cpu().detach().numpy())
            # vis.plot_imgs(detection_feature_[:, None, :, :],
            #              win=7, name='detection_feature')
            # vis.plot_img(detection_feature_[0:3, :, :], win=7, name='feature')
        kernel_cls = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num_per_position, 256, 4, 4)
        kernel_reg = self.conv_reg1(template_feature).view(N, 4 * self.anchor_num_per_position, 256, 4, 4)
        conv_cls = self.conv_cls2(detection_feature)
        conv_reg = self.conv_reg2(detection_feature)
        cls_map_size = list(conv_cls.shape)[-1]
        conv_cls = conv_cls.reshape(1, -1, cls_map_size, cls_map_size)   # 改变形状，才能进行两者卷积
        kernel_cls = kernel_cls.reshape(-1, 256, 4, 4)
        reg_map_size = list(conv_reg.shape)[-1]
        conv_reg = conv_reg.reshape(1, -1, reg_map_size, reg_map_size)
        kernel_reg = kernel_reg.reshape(-1, 256, 4, 4)
        pred_cls = F.conv2d(conv_cls, kernel_cls, groups=N)
        pred_reg = F.conv2d(conv_reg, kernel_reg, groups=N)
        return pred_cls, pred_reg


    def track_init(self, template):
        """
        :param template:  输入第一帧图片，把固定的值先计算好并且缓存，作为卷积核固定
        """
        N = template.size[0]
        template_feature = self.sharedFeatExtra(template)
        kernel_cls = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num_per_position, 256, 4, 4)
        kernel_reg = self.conv_reg1(template_feature).view(N, 4 * self.anchor_num_per_position, 256, 4, 4)
        self.kernel_cls = kernel_cls.reshape(-1, 256, 4, 4)
        self.kernel_reg = kernel_reg.reshape(-1, 256, 4, 4)


    def tracking(self, detection):
        N = detection.size(0)
        detection_feature = self.sharedFeatExtra(detection)
        conv_cls = self.conv_cls2(detection_feature)
        conv_reg = self.conv_reg2(detection_feature)
        cls_map_size = list(conv_cls.shape)[-1]
        conv_cls = conv_cls.reshape(1, -1, cls_map_size, cls_map_size)  # 改变形状，才能进行两者卷积
        reg_map_size = list(conv_reg.shape)[-1]
        conv_reg = conv_reg.reshape(1, -1, reg_map_size, reg_map_size)
        pred_cls = F.conv2d(conv_cls, self.kernel_cls, groups=N)
        pred_reg = F.conv2d(conv_reg, self.kernel_reg, groups=N)
        return pred_cls, pred_reg
