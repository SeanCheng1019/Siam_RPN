import numpy as np
from torch import nn
import torch as t
from net.config import Config
import torch.nn.functional as F
from net.config import Config
from net.vgg16 import VGG16
from net.stmm import STMM
#from net.alexnet import AlexNet
from lib.viusal import visual
from lib.util import norm_to_255


class SiameseAlexNet(nn.Module):
    def __init__(self, ):
        super(SiameseAlexNet, self).__init__()
        self.sharedFeatExtra = nn.Sequential(
            # conv1
            nn.Conv2d(3, 192, 11, stride=2),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(inplace=True),
            # conv2
            nn.Conv2d(192, 512, 5),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(inplace=True),
            # conv3
            nn.Conv2d(512, 768, 3),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(768, 768, 3),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(768, 512, 3),
            nn.BatchNorm2d(512),
        )
        if Config.update_template:
            # self.vgg16 = VGG16()
            # self.alexnet = AlexNet().cuda()
            self.alexnet = self.sharedFeatExtra().cuda()
            self.channel_adjust = nn.Conv2d(256, 512, 1)
            # load pretrained paramter to stmm module
            # pretrained_checkpoint = t.load(Config.pretrained_model)
            # pretrained_checkpoint = \
            #    {k.replace('features.features', 'sharedFeatExtra'): v for k, v in pretrained_checkpoint.items()}
            # model_dict = self.alexnet.state_dict()
            # model_dict.update(pretrained_checkpoint)
            # self.alexnet.load_state_dict(model_dict)
            # print("finish loading stmm_module's pre-trained model \n")
            # stmm module
            self.stmm = STMM(8, Config.his_window + 1, 512, 512, 1) # 加了1帧GT

        self.anchor_num_per_position = Config.anchor_num
        self.input_size = Config.instance_size
        self.conv_cls1 = nn.Conv2d(512, 512 * 2 * self.anchor_num_per_position, 3)
        self.conv_cls2 = nn.Conv2d(512, 512, 3)
        self.conv_reg1 = nn.Conv2d(512, 512 * 4 * self.anchor_num_per_position, 3)
        self.conv_reg2 = nn.Conv2d(512, 512, 3)
        self.regress_adjust = nn.Conv2d(4 * self.anchor_num_per_position,
                                        4 * self.anchor_num_per_position, 1)
        if Config.update_template:
            self.stmm_mem_adjust = nn.Conv2d(512, 256, 1)

    def forward(self, template, detection, his_frame, training=False):
        """
        :param template: torch.Size([48, 127, 127, 3]) 可正常传入图片 无重复
        :param detection:  torch.Size([8, 271, 271, 3]) 正常 不重复
        :param training:
        :param his_frame:  历史5帧
        :return:
        """
        N = template.size(0)
        if Config.update_template:
            template_wh = template.size(2)
            if training:
                templates = his_frame.view(Config.stmm_train_batch_size, Config.his_window, template_wh,
                                           template_wh, 3)
                # detection = detection.view(Config.stmm_train_batch_size, 3, detection_wh, detection_wh)

            else:
                templates = his_frame.view(Config.stmm_valid_batch_size, Config.his_window, template_wh,
                                           template_wh, 3)
                # detection = detection.view(Config.stmm_valid_batch_size, 3, detection_wh, detection_wh)

            template_stmm = \
                templates[:, 0:Config.his_window, :, :, :].permute(0, 1, 4, 2, 3).contiguous().view(-1,
                                                                                                    3,
                                                                                                    Config.exemplar_size,
                                                                                                    Config.exemplar_size
                                                                                                    ).float()

            # template = template.view(-1, 3, Config.exemplar_size, Config.exemplar_size)
            # detection = detection.view(-1, 3, Config.instance_size, Config.instance_size)
        template = template.permute(0, 3, 1, 2)
        detection = detection.permute(0, 3, 1, 2)

        template_feature = self.sharedFeatExtra(template)
        detection_feature = self.sharedFeatExtra(detection)
        if Config.update_template:
            # his_template_feature = self.vgg16(template_stmm.cuda())  # shape:[N, 512, 6, 6]
            his_template_feature = self.alexnet(template_stmm.cuda())  # shape:[N, 512, 6, 6]
            first_template_feature = self.alexnet(template)
            mixed_feature = t.cat([first_template_feature, his_template_feature])  # GT加上历史帧裁剪的模板
            his_mem = self.stmm(mixed_feature)
            update_mem_feature = his_mem[:, -1, :, :, :]  # shape:[N, 512, 6, 6]
            update_mem_feature = self.stmm_mem_adjust(update_mem_feature.cuda())
            # 历史5帧融合成的新模板特征，和第一帧的模板特征线性加权 （需要改进，线性加权） 同时这个要要加一个loss
            update_mem_feature = Config.template_combinition_coef * template_feature + (
                    1 - Config.template_combinition_coef) * update_mem_feature
            kernel_cls = self.conv_cls1(update_mem_feature).view(N, 2 * self.anchor_num_per_position, 256, 4, 4)
            kernel_reg = self.conv_reg1(update_mem_feature).view(N, 4 * self.anchor_num_per_position, 256, 4, 4)
        else:
            kernel_cls = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num_per_position, 512, 4, 4)
            kernel_reg = self.conv_reg1(template_feature).view(N, 4 * self.anchor_num_per_position, 512, 4, 4)
        conv_cls = self.conv_cls2(detection_feature)
        conv_reg = self.conv_reg2(detection_feature)
        cls_map_size = list(conv_cls.shape)[-1]
        conv_cls = conv_cls.reshape(1, -1, cls_map_size, cls_map_size)  # 改变形状，才能进行两者卷积
        kernel_cls = kernel_cls.reshape(-1, 512, 4, 4)
        reg_map_size = list(conv_reg.shape)[-1]
        conv_reg = conv_reg.reshape(1, -1, reg_map_size, reg_map_size)
        kernel_reg = kernel_reg.reshape(-1, 512, 4, 4)
        pred_cls = F.conv2d(conv_cls, kernel_cls, groups=N)
        pred_reg = F.conv2d(conv_reg, kernel_reg, groups=N)
        pred_reg = self.regress_adjust(pred_reg.reshape(N, 4 * self.anchor_num_per_position,
                                                        Config.train_map_size, Config.train_map_size))
        return pred_cls, pred_reg

    def track_init(self, template):
        """
        :param template:  输入第一帧图片，把固定的值先计算好并且缓存，作为卷积核固定
        """
        N = template.size(0)
        self.template_img = template
        template_feature = self.sharedFeatExtra(template)
        self.origin_template_feature = template_feature
        # self.first_template_feature = template_feature
        kernel_cls = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num_per_position, 512, 4, 4)
        kernel_reg = self.conv_reg1(template_feature).view(N, 4 * self.anchor_num_per_position, 512, 4, 4)
        self.kernel_cls = kernel_cls.reshape(-1, 512, 4, 4)
        self.kernel_reg = kernel_reg.reshape(-1, 512, 4, 4)

    def track_update_template(self, his_templates):
        """
        :param his_templates : as list type  给定的输入是历史帧5张
        :return: self.kernel_cls, self.kernel_reg
        """
        N = his_templates[0].size(0)
        his_templates = [x.numpy() for x in his_templates]
        his_templates = np.stack(his_templates).transpose(1, 0, 2, 3, 4)
        his_templates = t.from_numpy(his_templates)
        template_stmm = \
            his_templates[0:Config.his_window, :, :, :].permute(0, 1, 4, 2, 3).contiguous().view(-1,
                                                                                                 3,
                                                                                                 Config.exemplar_size,
                                                                                                 Config.exemplar_size
                                                                                                 ).float()
        his_template_feature = self.alexnet(template_stmm.cuda())  # shape:[5, 512, 6, 6]
        self.first_template_feature = self.alexnet(self.template_img.cuda())
        mixed_feature = t.cat([self.first_template_feature, his_template_feature])
        his_mem = self.stmm(mixed_feature)
        update_mem_feature = his_mem[:, -1, :, :, :]  # shape:[N, 512, 6, 6]
        update_mem_feature = self.stmm_mem_adjust(update_mem_feature.cuda())
        # 历史5帧融合成的新模板特征，和第一帧的模板特征线性加权
        new_template_feature = Config.template_combinition_coef * self.origin_template_feature + (
                1 - Config.template_combinition_coef) * update_mem_feature
        self.kernel_cls = self.conv_cls1(new_template_feature).view(N, 2 * self.anchor_num_per_position, 256, 4,
                                                                    4).reshape(-1, 256, 4, 4)
        self.kernel_reg = self.conv_reg1(new_template_feature).view(N, 4 * self.anchor_num_per_position, 256, 4,
                                                                    4).reshape(-1, 256, 4, 4)

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
        map_size = pred_reg.size()[-1]
        pred_reg = self.regress_adjust(pred_reg.reshape(N, 4 * self.anchor_num_per_position,
                                                        map_size, map_size))
        return pred_cls, pred_reg
