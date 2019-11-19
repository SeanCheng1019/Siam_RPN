from net.net_siamrpn import SiameseAlexNet
from torchvision.transforms import transforms
from lib.custom_transforms import ToTensor
from lib.util import generate_anchors, get_exemplar_img, get_instance_img, box_transform_use_reg_offset
from net.config import Config
import numpy as np
import torch.nn.functional as F


class SiamRPNTracker:
    def __init__(self, model_path):
        self.model = SiameseAlexNet()
        self.model = self.model.cuda()
        self.model.eval()
        self.transforms = transforms.Compose([
            ToTensor()
        ])
        valid_map_size = Config.valid_map_size
        self.anchors = generate_anchors(total_stride=Config.total_stride,
                                        base_size=Config.anchor_base_size,
                                        scales=Config.anchor_scales,
                                        ratios=Config.anchor_ratio,
                                        score_map_size=Config.score_map_size)

    def init(self, frame, bbox):
        self.center_pos = np.array(  # cx,cy
            bbox[0] + bbox[2] / 2 - 0.5, bbox[1] + bbox[3] / 2 - 0.5
        )
        self.target_sz = np.array([bbox[2], bbox[3]])  # w,h
        self.target_sz_w, self.target_sz_h = bbox[2], bbox[3]
        self.origin_target_sz = self.target_sz.copy()
        self.bbox = np.hstack(self.center_pos.copy(), self.target_sz.copy())  # cx, cy, w,h
        self.img_mean = np.mean(frame, axis=(0, 1))
        exemplar_img, scale_ratio = get_exemplar_img(frame, self.bbox, Config.exemplar_size,
                                                     Config.context_margin_amount, self.img_mean)
        exemplar_img = self.transforms(exemplar_img)[None, :, :, :]
        self.model.track_init(exemplar_img.permute(0, 3, 1, 2).cuda())

    def update(self, frame):
        """
        :param frame: an RGB image
        :return: bbox: [xmin, ymin, xmax, ymax]
        """
        instance_img, _, _, scale_detection = get_instance_img(frame, self.bbox, Config.exemplar_size,
                                                               Config.instance_size, Config.context_margin_amount,
                                                               self.img_mean)
        instance_img = self.transforms(instance_img)[None, :, :, :]
        pred_cls, pred_reg = self.model.tracking(detection=instance_img.permute(0, 3, 1, 2).cuda())
        pred_cls = pred_cls.reshape(-1, 2,
                                    Config.anchor_num * Config.score_map_size * Config.score_map_size).permute(0,
                                                                                                               2,
                                                                                                               1)
        pred_reg = pred_reg.reshape(-1, 4,
                                    Config.anchor_num * Config.score_map_size * Config.score_map_size).permute(0,
                                                                                                               2,
                                                                                                               1)
        # 预测的dx,dy,dw,dh
        delta = pred_reg.cpu().detach().numpy()
        pred_box = box_transform_use_reg_offset(self.anchors, delta)
        pred_score = F.softmax(pred_cls, dim=2)[0, :, 1].cpu().detach().numpy()  # softmax的含义

        # 以下内容在论文里4.3节中有解释
        def max_ratio(ratio):
            # ratio represents the proposal’s ratio of height and width
            return np.maximum(ratio, 1 / ratio)

        def overall_scale(w, h):
            pad = (w + h) * 1 / 2
            # s represent the overall scale of the proposal
            s = (w + pad) * (h + pad)
            return np.sqrt(s)

        # 惩罚和加窗
        s = max(overall_scale(pred_box[:, 2], pred_box[:, 3]) /
                # 乘以scale_detection的原因：pred_box是放缩后的图上预测出的box，而target_sz_w/h是原图的尺寸，两者要同步尺寸
                (overall_scale(self.target_sz_w * scale_detection, self.target_sz_h * scale_detection)))
        r = max((self.target_sz_w / self.target_sz_h) / (pred_box[:, 2] / pred_box[:, 3]))
        penalty = np.exp(-(r * s - 1) * Config.penalty_k)
        pred_score = pred_score * penalty  #加惩罚
        pred_score = pred_score * (1- Config.window_influence) + self.  # 加余弦窗
        highest_score_id = np.argmax(pred_score)
        target = pred_box[highest_score_id, :] / scale_detection  # 返回原图要用原来的尺寸
        lr =
        # 预测的x，y都是相对于中心点的相对位移
        new_x = np.clip(target[0] + self.center_pos[0], 0, frame.shape[1])
        new_y = np.clip(target[1] + self.center_pos[1], 0, frame.shape[0])
        new_w = np.clip(self.)
