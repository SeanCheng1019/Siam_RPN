from net.net_siamrpn import SiameseAlexNet
from torchvision.transforms import transforms
from lib.custom_transforms import ToTensor
from lib.util import generate_anchors, get_exemplar_img, get_instance_img, box_transform_use_reg_offset, compute_iou
from net.config import Config
import numpy as np
import torch.nn.functional as F
import torch as t
from got10k.trackers import Tracker
from lib.util import use_others_model
import cv2
#from bin.network import SiameseAlexNet

class SiamRPNTracker(Tracker):
    def __init__(self, model_path):
        super(SiamRPNTracker, self).__init__(
            name='SiamRPN', is_deterministic=True
        )
        self.model = SiameseAlexNet()
        use_others = Config.use_others
        checkpoint = t.load(model_path)
        if use_others:
            checkpoint = use_others_model(checkpoint)
        print("-------------------loading trained model-----------------------\n")
        if 'model' in checkpoint.keys():
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
        print("------------------------finishing loading-----------------------\n")
        self.model = self.model.cuda()
        self.model.eval()
        self.transforms = transforms.Compose([
            ToTensor()
        ])
        self.update_ = False
        self.frame_count = 0
        self.template_save_num_count = 0
        if Config.update_template:
            self.his_frame_pool = []
        if Config.select_template:
            assert Config.select_template == Config.update_template, \
                "you must update template then you can select different template"

    def init(self, frame, bbox):
        """
        :param frame: current frame
        :param bbox: left top corner, w, h
        :return:
        """
        frame = np.array(frame)

        # change to [cx,cy,w,h]
        bbox = np.array([
            bbox[0] + bbox[2] / 2 - 0.5,
            bbox[1] + bbox[3] / 2 - 0.5,
            bbox[2],
            bbox[3]]
        )
        self.center_pos = bbox[:2]
        self.target_sz = bbox[2:]
        replace = False
        # for small target, use larger search region
        if np.prod(self.target_sz) / np.prod(frame.shape[:2]) < 0.004:
            replace = True
        # 可以提前计算好响应图的尺寸大小，用来生成anchors
        self.instance_size = Config.instance_size if not replace else 287
        self.valid_map_size = int((self.instance_size - Config.exemplar_size) / Config.total_stride) + 1
        self.anchors = generate_anchors(total_stride=Config.total_stride,
                                        base_size=Config.anchor_base_size,
                                        scales=Config.anchor_scales,
                                        ratios=Config.anchor_ratio,
                                        score_map_size=self.valid_map_size)
        hanning = np.hanning(self.valid_map_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), Config.anchor_num)
        self.target_sz_w, self.target_sz_h = bbox[2], bbox[3]
        self.origin_target_sz = np.array([bbox[2], bbox[3]])  # w, h
        self.img_mean = np.mean(frame, axis=(0, 1))
        exemplar_img, scale_ratio, _ = get_exemplar_img(frame, bbox, Config.exemplar_size,
                                                        Config.context_margin_amount, self.img_mean)
        # save exemplar image to the disk for observe
        f_name = "/media/csy/62ac73e0-814c-4dba-b59d-676690aca14b/experiments pictures/siamrpn_templates/{}.jpeg".format(self.template_save_num_count)
        self.template_save_num_count += 1
        cv2.imwrite(f_name, exemplar_img)

        exemplar_img = self.transforms(exemplar_img)[None, :, :, :]
        if Config.update_template:
            self.frame_count += 1
            # self.his_frame_pool.append(exemplar_img)
        self.model.track_init(exemplar_img.permute(0, 3, 1, 2).cuda())


    def update(self, frame):
        """
        :param frame: an numpy type image with 3 channel
        :return: bbox：[xmin, ymin, w, h]
        """
        frame = np.array(frame)
        self.img_mean = np.mean(frame, axis=(0, 1))
        crop_box = None
        box = np.hstack([self.center_pos, self.target_sz])
        if Config.update_template:
            # 保存的历史帧溢出 （超过规定的窗口数量） 始终保存最新的5帧
            if self.his_frame_pool.__len__() > 5:
                del (self.his_frame_pool[0])
                # self.frame_count = 5
            if self.his_frame_pool.__len__() == 5:
                # 更新模板 更新两个原先由第一帧固定的卷积核
                # print("更新模板\n")
                # 隔五帧更新
                if (self.frame_count - 1) % 5 == 0 and self.frame_count != 1:
                    self.update_ = True
                else:
                    self.update_ = False
                if self.update_:
                    self.model.track_update_template(his_templates=self.his_frame_pool)

        instance_img, _, _, scale_detection = get_instance_img(frame, box, Config.exemplar_size,
                                                               self.instance_size, Config.context_margin_amount,
                                                               self.img_mean)
        instance_img = self.transforms(instance_img)[None, :, :, :]
        if Config.select_template and self.update_:  # need choose template
            pred_cls, pred_reg, pred_cls_ori, pred_reg_ori = self.model.tracking(detection=instance_img.permute(0, 3, 1, 2).cuda(), update_=True)
            pred_reg_ori = pred_reg_ori.reshape(-1, 4,
                                        Config.anchor_num * self.valid_map_size * self.valid_map_size).permute(0,
                                                                                                               2,
                                                                                                               1)
        else:  # 1.update template but don't choose template 2. don't update template
            pred_cls, pred_reg = self.model.tracking(detection=instance_img.permute(0, 3, 1, 2).cuda())
        # pred_cls = pred_cls.reshape(-1, 2,
        #                             Config.anchor_num * Config.score_map_size * Config.score_map_size).permute(0,
        #                                                                                                        2,
        #                                                                                                        1)
        pred_reg = pred_reg.reshape(-1, 4,
                                    Config.anchor_num * self.valid_map_size * self.valid_map_size).permute(0,
                                                                                                           2,
                                                                                                           1)
        # 预测的dx,dy,dw,dh
        delta = pred_reg.cpu().detach().numpy().squeeze()
        pred_box = box_transform_use_reg_offset(self.anchors, delta).squeeze()
        pred_score = F.softmax(pred_cls.permute(
            1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1].cpu().numpy()
        if Config.update_template and Config.select_template and self.update_:
            delta_ori = pred_reg_ori.cpu().detach().numpy().squeeze()
            pred_box_ori = box_transform_use_reg_offset(self.anchors, delta_ori).squeeze()
            pred_score_ori = F.softmax(pred_cls_ori.permute(
                1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1].cpu().numpy()

        # 以下内容在论文里4.3节中有解释
        def max_ratio(ratio):
            # ratio represents the proposal’s ratio of height and width
            return np.maximum(ratio, 1.0 / ratio)

        def overall_scale(w, h):
            pad = (w + h) * 1 / 2
            # s represent the overall scale of the proposal
            s = (w + pad) * (h + pad)
            return np.sqrt(s)

        # 惩罚和加窗
        s = max_ratio(overall_scale(pred_box[:, 2], pred_box[:, 3]) /
                      # 乘以scale_detection的原因：pred_box是放缩后的图上预测出的box，而target_sz_w、_h是原图的尺寸，两者要同步尺寸
                      (overall_scale(self.target_sz_w * scale_detection, self.target_sz_h * scale_detection)))
        r = max_ratio((self.target_sz_w / self.target_sz_h) / (pred_box[:, 2] / pred_box[:, 3]))
        penalty = np.exp(-(r * s - 1) * Config.penalty_k)  # 对大尺度的变化要惩罚，让其权重配小权重
        pred_score = pred_score * penalty  # 加惩罚
        pred_score = pred_score * (1 - Config.window_influence) + \
                     self.window * Config.window_influence  # 加余弦窗
        highest_score_id = np.argmax(pred_score)
        target = pred_box[highest_score_id, :] / scale_detection  # 返回原图要用原来的尺寸
        lr = pred_score[highest_score_id] * Config.track_lr
        # 预测的x，y都是相对于中心点的相对位移
        # clip boundary  （准备用一个clip函数统一处理来替代下面步骤）
        # smooth bbox ，不是完全使用预测出来的新bbox，而是有个平滑的变化，类似于在之前长度上的一个增量。
        res_x = np.clip(target[0] + self.center_pos[0], 0, frame.shape[1])
        res_y = np.clip(target[1] + self.center_pos[1], 0, frame.shape[0])
        res_w = np.clip(self.target_sz_w * (1 - lr) + target[2] * lr,
                        Config.min_scale * self.origin_target_sz[0],
                        Config.max_scale * self.origin_target_sz[0])
        res_h = np.clip(self.target_sz_h * (1 - lr) + target[3] * lr,
                        Config.min_scale * self.origin_target_sz[1],
                        Config.max_scale * self.origin_target_sz[1])
        # update state
        if (not Config.select_template) or (Config.select_template and not self.update_):
            self.center_pos = np.array([res_x, res_y])
            self.target_sz_w = res_w
            self.target_sz_h = res_h
            self.target_sz = np.array([res_w, res_h])
        bbox = np.array([res_x, res_y, res_w, res_h])
        # 转换为left-top 来可视化画图
        box = np.array([
            np.clip(bbox[0] - bbox[2] / 2, 0, frame.shape[1]).astype(np.float64),
            np.clip(bbox[1] - bbox[3] / 2, 0, frame.shape[0]).astype(np.float64),
            np.clip(bbox[2], 10, frame.shape[1]).astype(np.float64),
            np.clip(bbox[3], 10, frame.shape[0]).astype(np.float64)
        ])
        if Config.update_template and Config.select_template and self.update_:
            s_ori = max_ratio(overall_scale(pred_box_ori[:, 2], pred_box_ori[:, 3]) /
                          # 乘以scale_detection的原因：pred_box是放缩后的图上预测出的box，而target_sz_w、_h是原图的尺寸，两者要同步尺寸
                          (overall_scale(self.target_sz_w * scale_detection, self.target_sz_h * scale_detection)))
            r_ori = max_ratio((self.target_sz_w / self.target_sz_h) / (pred_box_ori[:, 2] / pred_box_ori[:, 3]))
            penalty_ori = np.exp(-(r_ori * s_ori - 1) * Config.penalty_k)  # 对大尺度的变化要惩罚，让其权重配小权重
            pred_score_ori = pred_score_ori * penalty_ori  # 加惩罚
            pred_score_ori = pred_score_ori * (1 - Config.window_influence) + \
                         self.window * Config.window_influence  # 加余弦窗
            highest_score_id_ori = np.argmax(pred_score_ori)
            target_ori = pred_box_ori[highest_score_id_ori, :] / scale_detection  # 返回原图要用原来的尺寸
            lr_ori = pred_score_ori[highest_score_id_ori] * Config.track_lr
            # 预测的x，y都是相对于中心点的相对位移
            # clip boundary  （准备用一个clip函数统一处理来替代下面步骤）
            # smooth bbox ，不是完全使用预测出来的新bbox，而是有个平滑的变化，类似于在之前长度上的一个增量。
            res_x_ori = np.clip(target_ori[0] + self.center_pos[0], 0, frame.shape[1])
            res_y_ori = np.clip(target_ori[1] + self.center_pos[1], 0, frame.shape[0])
            res_w_ori = np.clip(self.target_sz_w * (1 - lr_ori) + target_ori[2] * lr_ori,
                            Config.min_scale * self.origin_target_sz[0],
                            Config.max_scale * self.origin_target_sz[0])
            res_h_ori = np.clip(self.target_sz_h * (1 - lr_ori) + target[3] * lr_ori,
                            Config.min_scale * self.origin_target_sz[1],
                            Config.max_scale * self.origin_target_sz[1])
            bbox_ori = np.array([res_x_ori, res_y_ori, res_w_ori, res_h_ori])
            box_ori = np.array([
                np.clip(bbox_ori[0] - bbox_ori[2] / 2, 0, frame.shape[1]).astype(np.float64),
                np.clip(bbox_ori[1] - bbox_ori[3] / 2, 0, frame.shape[0]).astype(np.float64),
                np.clip(bbox_ori[2], 10, frame.shape[1]).astype(np.float64),
                np.clip(bbox_ori[3], 10, frame.shape[0]).astype(np.float64)
            ])

            # compute two box IOU to select template
            iouOf2temlpates = compute_iou(bbox, bbox_ori)
            if iouOf2temlpates >= 0.6:
                # update target state for the following tracking procedure
                self.center_pos = np.array([res_x_ori, res_y_ori])
                self.target_sz_w = res_w_ori
                self.target_sz_h = res_h_ori
                self.target_sz = np.array([res_w_ori, res_h_ori])
                crop_box = bbox_ori
            else:
                self.center_pos = np.array([res_x, res_y])
                self.target_sz_w = res_w
                self.target_sz_h = res_h
                self.target_sz = np.array([res_w, res_h])
                crop_box = bbox
        else:
            crop_box = box
        # save history image
        if Config.update_template:
            croped_template, _, _ = get_exemplar_img(frame, crop_box, Config.exemplar_size,
                                                     Config.context_margin_amount,
                                                     self.img_mean)
            # 将历史帧裁剪后存到模板池中
            # save history template image to the disk for observe
            f_name = "/media/csy/62ac73e0-814c-4dba-b59d-676690aca14b/experiments pictures/siamrpn_templates/{}.jpeg".format(
                self.template_save_num_count)
            self.template_save_num_count += 1
            cv2.imwrite(f_name, croped_template)
            croped_template = self.transforms(croped_template)[None, :, :, :]
            self.his_frame_pool.append(croped_template)
            self.frame_count += 1
        if Config.update_template and Config.select_template and self.update_:
            return box_ori
        else:
            return box
