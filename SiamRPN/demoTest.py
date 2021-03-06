from net.net_siamrpn import SiameseAlexNet
from torchvision.transforms import transforms
from lib.custom_transforms import ToTensor
from lib.util import generate_anchors, get_exemplar_img, get_instance_img, box_transform_use_reg_offset
from net.config import Config
import numpy as np
import torch.nn.functional as F
import torch as t
from got10k.trackers import Tracker
from lib.util import use_others_model

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
        # 可以提前计算好响应图的尺寸大小，用来生成anchors
        valid_map_size = Config.valid_map_size
        self.anchors = generate_anchors(total_stride=Config.total_stride,
                                        base_size=Config.anchor_base_size,
                                        scales=Config.anchor_scales,
                                        ratios=Config.anchor_ratio,
                                        score_map_size=valid_map_size)
        hanning = np.hanning(valid_map_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), Config.anchor_num)

    def init(self, frame, bbox):
        """
        :param frame: current frame
        :param bbox: left top corner, w, h
        :return:
        """
        # change to [cx,cy,w,h]
        bbox = np.array([
            bbox[0] + bbox[2] / 2 - 0.5,
            bbox[1] + bbox[3] / 2 - 0.5,
            bbox[2],
            bbox[3]]
        )
        self.center_pos = bbox[:2]
        self.target_sz = bbox[2:]
        self.target_sz_w, self.target_sz_h = bbox[2], bbox[3]
        self.origin_target_sz = np.array([bbox[2], bbox[3]])  # w, h
        self.img_mean = np.mean(frame, axis=(0, 1))
        frame = np.array(frame)
        exemplar_img, scale_ratio, _ = get_exemplar_img(frame, bbox, Config.exemplar_size,
        Config.context_margin_amount, self.img_mean)
        exemplar_img = self.transforms(exemplar_img)[None, :, :, :]
        self.model.track_init(exemplar_img.permute(0, 3, 1, 2).cuda())


    def update(self, frame):
        """
        :param frame: an numpy type image with 3 channel
        :return: bbox：[xmin, ymin, w, h]
        """
        frame = np.array(frame)
        box = np.hstack([self.center_pos, self.target_sz])
        self.img_mean = np.mean(frame, axis=(0, 1))
        instance_img, _, _, scale_detection = get_instance_img(frame, box, Config.exemplar_size,
                                                               Config.instance_size, Config.context_margin_amount,
                                                               self.img_mean)
        instance_img = self.transforms(instance_img)[None, :, :, :]
        pred_cls, pred_reg = self.model.tracking(detection=instance_img.permute(0, 3, 1, 2).cuda())
        # pred_cls = pred_cls.reshape(-1, 2,
        #                             Config.anchor_num * Config.score_map_size * Config.score_map_size).permute(0,
        #                                                                                                        2,
        #                                                                                                        1)
        pred_reg = pred_reg.reshape(-1, 4,
                                    Config.anchor_num * Config.valid_map_size * Config.valid_map_size).permute(0,
                                                                                                               2,
                                                                                                               1)
        # 预测的dx,dy,dw,dh
        delta = pred_reg.cpu().detach().numpy().squeeze()
        pred_box = box_transform_use_reg_offset(self.anchors, delta).squeeze()
        pred_score = F.softmax(pred_cls.permute(
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
        lr = penalty[highest_score_id] * pred_score[highest_score_id] * Config.track_lr
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
        self.center_pos = np.array([res_x, res_y])
        self.target_sz = np.array([res_w, res_h])
        bbox = np.array([res_x, res_y, res_w, res_h])
        # 转换为left-top 来可视化画图
        box = np.array([
            np.clip(bbox[0] - bbox[2] / 2, 0, frame.shape[1]).astype(np.float64),
            np.clip(bbox[1] - bbox[3] / 2, 0, frame.shape[0]).astype(np.float64),
            np.clip(bbox[2], 10, frame.shape[1]).astype(np.float64),
            np.clip(bbox[3], 10, frame.shape[0]).astype(np.float64)
        ])


        # return self.box, pred_score[highest_score_id]
        return box
