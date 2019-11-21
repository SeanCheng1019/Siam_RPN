from net.config import Config
from net.net_siamrpn import SiameseAlexNet
from lib.tracker import SiamRPNTracker
from data.vot import VOTDataset
from lib.util import get_axis_aligned_box
import numpy as np
import cv2


def main(model_path, dataset_path, dataset_name, vis=True):
    tracker = SiamRPNTracker(model_path)
    # loading testing dataset
    dataset = VOTDataset(name=dataset_name, datasets_path=dataset_path, load_img=False)
    model_name = model_path.split('/')[-1].split('.')[0]
    total_lost = 0
    for v_idx, video in enumerate(dataset):
        frame_counter = 0
        lost_number = 0
        toc = 0
        pred_bboxes = []
        for idx, (img, gt_bbox) in enumerate(video):
            if len(gt_bbox) == 4:  # VOT的gt有8个点 不是平行于坐标轴的，可以是任意摆放的矩形框
                gt_bbox = [gt_bbox[0], gt_bbox[1], gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                           gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                           gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
            tic = cv2.getTickCount()
            if idx == frame_counter:
                cx, cy, w, h = get_axis_aligned_box(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                pred_bboxes.append(1)
            elif idx > frame_counter:
                output = tracker.update(img)
                pred_bbox = output[0]
                overlap = vot


if __name__ == '__main__':
    testing_datasets = ''
    main()
