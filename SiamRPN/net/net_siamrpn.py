import numpy as np
from torch import nn
from net.config import Config
import torch.nn.functional as F
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
        #self.conv_displacement = int((self.input_size - Config.instance_size) / Config.total_stride)
        self.conv_cls1 = nn.Conv2d(256, 256 * 2 * self.anchor_num_per_position, 3)
        self.conv_cls2 = nn.Conv2d(256, 256, 3)
        self.conv_reg1 = nn.Conv2d(256, 256 * 4 * self.anchor_num_per_position, 3)
        self.conv_reg2 = nn.Conv2d(256, 256, 3)

    def forward(self, template, detection):
        N = template.size(0)
        template_feature = self.sharedFeatExtra(template)
        detection_feature = self.sharedFeatExtra(detection)
        kernel_cls = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num_per_position, 256, 4, 4)
        kernel_reg = self.conv_reg1(template_feature).view(N, 4 * self.anchor_num_per_position, 256, 4, 4)
        conv_cls = self.conv_cls2(detection_feature)
        conv_reg = self.conv_reg2(detection_feature)

        kernel_cls = kernel_cls.reshape(-1, 256, 4, 4)
        kernel_reg = kernel_reg.reshape(-1, 256, 4, 4)
        #conv_cls = conv_cls.reshape(1, -1, self.conv_displacement + 4, self.conv_displacement + 4)
        #conv_reg = conv_reg.reshape(1, -1, self.conv_displacement + 4, self.conv_displacement + 4)

        pred_cls = F.conv2d(conv_cls, kernel_cls, groups=N)
        pred_reg = F.conv2d(conv_reg, kernel_reg, groups=N)
        return pred_cls, pred_reg



