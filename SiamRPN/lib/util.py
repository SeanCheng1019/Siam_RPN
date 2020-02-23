import numpy as np
import cv2
import torch as t
from net.config import Config


def get_center(x):
    return (x - 1.0) / 2


def xyxy2cxcywh(bbox):
    return get_center(bbox[0] + bbox[2]), \
           get_center(bbox[1] + bbox[3]), \
           (bbox[2] - bbox[0]), \
           (bbox[3] - bbox[1])


def cxcywh2xyxy(bboxes):
    if len(np.array(bboxes).shape) == 1:
        bboxes = np.array(bboxes)[None, :]
    else:
        bboxes = np.array(bboxes)
    x1 = bboxes[:, 0:1] - bboxes[:, 2:3] / 2 + 0.5
    x2 = x1 + bboxes[:, 2:3] - 1
    y1 = bboxes[:, 1:2] - bboxes[:, 3:4] / 2 + 0.5
    y2 = y1 + bboxes[:, 3:4] - 1
    return np.concatenate([x1, y1, x2, y2], 1), x1, x2, y1, y2


def get_instance_img(img, bbox, size_z, size_x, context_margin_amount, img_mean=None):
    cx, cy, w, h = bbox
    w_context = w + context_margin_amount * (w + h)
    h_context = h + context_margin_amount * (w + h)
    # 还没缩放尺寸前的exemplar patch size， 是个正方形patch，但是尺寸还没缩放到127/255之类的
    size_original_exemplar = np.sqrt(w_context * h_context)
    # 这个ratio是两个正方形图片块缩放的比例，缩放前是原图上裁剪出的一块正方形，缩放后是127*127的一块正方形。
    scale_ratio = size_z / size_original_exemplar
    size_original_instance = size_original_exemplar * (size_x / size_z)
    instance_img, scale_ratio_x = crop_and_pad(img, cx, cy, size_x, size_original_instance, img_mean)
    # 因为resize过，所以目标的尺寸要从原图乘上resize的ratio
    w_instance = w * scale_ratio
    h_instance = h * scale_ratio
    return instance_img, w_instance, h_instance, scale_ratio


def get_exemplar_img(img, bbox, size_z, context_margin_amount, img_mean=None):
    cx, cy, w, h = bbox
    w_context = w + context_margin_amount * (w + h)
    h_context = h + context_margin_amount * (w + h)
    size_original_exemplar = np.sqrt(w_context * h_context)
    scale_ration_z = size_z / size_original_exemplar
    exemplar_img, _ = crop_and_pad(img, cx, cy, size_z, size_original_exemplar, img_mean)
    return exemplar_img, scale_ration_z, size_original_exemplar


def crop_and_pad(img, cx, cy, model_size, original_exemplar_size, img_mean=None):
    """
    :param img:
    :param cx:
    :param cy:
    :param model_size: 最终想要resize的尺寸
    :param original_exemplar_size: 希望裁剪出来的patch尺寸
    :param img_mean:
    :return:
    """
    im_h, im_w, im_c = img.shape
    xmin = cx - (original_exemplar_size - 1) / 2
    xmax = cx + (original_exemplar_size - 1) / 2
    ymin = cy - (original_exemplar_size - 1) / 2
    ymax = cy + (original_exemplar_size - 1) / 2
    # 各边需要填充的像素，最少填充0
    left_pad = int(round_up(max(0., -xmin)))
    top_pad = int(round_up(max(0., -ymin)))
    right_pad = int(round_up(max(0., xmax - im_w + 1)))
    bottom_pad = int(round_up(max(0., ymax - im_h + 1)))
    # 要同时考虑超过原图和没超过原图的情况，超过原图如上要补填充，没超过原图如下要在原图作裁剪
    # 最小从0开始
    # 此处的xmin ymin xmax ymax，是填充后，原图所在区域的界限
    xmin = int(round_up(xmin + left_pad))
    xmax = int(round_up(xmax + left_pad))
    ymin = int(round_up(ymin + top_pad))
    ymax = int(round_up(ymax + top_pad))
    # 开始填充
    if any([left_pad, right_pad, top_pad, bottom_pad]):

        tmp_img = np.zeros((im_h + top_pad + bottom_pad, im_w + left_pad + right_pad, im_c), np.uint8)
        # 把原图的值赋上
        tmp_img[top_pad:top_pad + im_h, left_pad:left_pad + im_w, :] = img
        if top_pad:
            tmp_img[0:top_pad, left_pad:left_pad + im_w, :] = img_mean
        if bottom_pad:
            tmp_img[im_h + top_pad:, left_pad:left_pad + im_w, :] = img_mean
        if left_pad:
            tmp_img[:, 0:left_pad, :] = img_mean
        if right_pad:
            tmp_img[:, im_w + left_pad:, :] = img_mean
        # 带了填充后的裁剪
        img_patch_original = tmp_img[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]  # 尺寸有问题， w、h的长度不一样，多了1
    else:
        # 无填充的情况下，直接在原图上裁剪
        img_patch_original = img[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
    if not np.array_equal(model_size, original_exemplar_size):  # resize到config里定义的尺寸。
        img_patch = cv2.resize(img_patch_original, (model_size, model_size))
    else:
        img_patch = img_patch_original
    scale_ratio = model_size / img_patch_original.shape[0]  # 这里img_path_original的两个尺寸长度不一样
    return img_patch, scale_ratio


def round_up(value):
    # 保证两位小数的精确四舍五入
    return round(value + 1e-6 + 1000) - 1000


def get_axis_aligned_box(region):
    """
        转换为cx,cy,w,h 平行于坐标轴形式的box
    """
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
             np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w / 2
        cy = y + h / 2
    return cx, cy, w, h


def box_delta_in_gt_anchor(anchors, gt_box):
    # 这里gt_box的cx，cy是相对位置，相对于中心点
    anchor_cx = anchors[:, :1]
    anchor_cy = anchors[:, 1:2]
    anchor_w = anchors[:, 2:3]
    anchor_h = anchors[:, 3:]
    gt_cx, gt_cy, gt_w, gt_h = gt_box
    # 论文里的公式
    delta_x = (gt_cx - anchor_cx) / anchor_w
    delta_y = (gt_cy - anchor_cy) / anchor_h
    delta_w = np.log(gt_w / anchor_w)
    delta_h = np.log(gt_h / anchor_h)
    # 返回原始锚框和gt_box的真正的"距离"，作为regression的GT
    regression_target = np.hstack((delta_x, delta_y, delta_w, delta_h))
    return regression_target


def box_transform_use_reg_offset(anchors, offsets):
    """
    用预测的偏移量计算出box的cx,cy,w,h
    """
    anchor_cx = anchors[:, :1]
    anchor_cy = anchors[:, 1:2]
    anchor_w = anchors[:, 2:3]
    anchor_h = anchors[:, 3:]
    offsets_x, offsets_y, offsets_w, offsets_h = offsets[:, :1], offsets[:, 1:2], \
                                                 offsets[:, 2:3], offsets[:, 3:]
    box_cx = anchor_w * offsets_x + anchor_cx
    box_cy = anchor_h * offsets_y + anchor_cy
    box_w = anchor_w * np.exp(offsets_w)   # will occur 'overflow encountered in exp'
    box_h = anchor_h * np.exp(offsets_h)
    boxes = np.stack([box_cx, box_cy, box_w, box_h], axis=2)
    return boxes


def compute_iou(anchors, box):
    # 为了计算的时候维度匹配
    if np.array(anchors).ndim == 1:
        anchors = np.array(anchors)[None, :]
    else:
        anchors = np.array(anchors)
    if np.array(box).ndim == 1:
        box = np.array(box)[None, :]
    else:
        box = np.array(box)
    # 将gt_box的个数复制到和锚框的个数一样
    gt_box = np.tile(box.reshape(1, -1), (anchors.shape[0], 1))
    _, anchor_x1, anchor_x2, anchor_y1, anchor_y2 = cxcywh2xyxy(anchors)
    _, gt_x1, gt_x2, gt_y1, gt_y2 = cxcywh2xyxy(gt_box)
    overlap_x1 = np.max([anchor_x1, gt_x1], axis=0)
    overlap_x2 = np.min([anchor_x2, gt_x2], axis=0)
    overlap_y1 = np.max([anchor_y1, gt_y1], axis=0)
    overlap_y2 = np.min([anchor_y2, gt_y2], axis=0)
    # 要注意到没有交集的情况，x2-x1就会出现负数，就没有意义，所以要限制最小是0
    overlap_area = np.max([(overlap_x2 - overlap_x1), np.zeros(overlap_x1.shape)], axis=0) \
                   * np.max([(overlap_y2 - overlap_y1), np.zeros(overlap_x1.shape)], axis=0)
    anchor_area = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    iou = overlap_area / (anchor_area + gt_area - overlap_area + 1e-6)
    return iou


def nms(bboxes, scores, num, threshold=0.7):
    '''
    先找到得分最大的框，然后其他框和这个最大的分的框计算iou，超过0.7的框都去除，其他的保留。
    '''
    sort_index = np.argsort(scores)[::-1]
    ordered_bboxes = bboxes[sort_index]
    selected_bbox = [ordered_bboxes[0]]
    selected_index = [sort_index[0]]
    for i, bbox in enumerate(ordered_bboxes):
        iou = compute_iou(selected_bbox, bbox)
        if np.max(iou) < threshold:
            selected_bbox.append(bbox)
            selected_index.append(sort_index[i])
            if len(selected_bbox) >= num:
                break
    return selected_index


def ajust_learning_rate(optimizer, decay=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


def generate_anchors(total_stride, base_size, scales, ratios, score_map_size):
    """
        anchors: cx,cy,w,h  这里的cx cy 是相对于图片中心的相对位置
    """
    anchor_num = len(scales) * len(ratios)  # 每个位置的锚框数量
    anchor = np.zeros((anchor_num, 4), dtype=np.float32)
    size = np.square(base_size)
    count = 0
    '''
    原来的面积是 base_size*base_size=size, 改变后是base_size0*base_size0*ratio=size
    所以计算新的边长是sqrt(size/ratio)
    '''
    for ratio in ratios:
        w_scaled0 = int(np.sqrt(size / ratio))
        h_scaled0 = int(w_scaled0 * ratio)
        for scale in scales:
            w_scaled = w_scaled0 * scale
            h_scaled = h_scaled0 * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = w_scaled
            anchor[count, 3] = h_scaled
            count += 1
    # 把一个位置上的anchor配置复制到所有位置上
    anchor = np.tile(anchor, score_map_size * score_map_size).reshape(-1, 4)
    shift = (score_map_size // 2) * total_stride
    xx, yy = np.meshgrid([-shift + total_stride * x for x in range(score_map_size)],
                         [-shift + total_stride * y for y in range(score_map_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    # print("finish generate anchors\n")
    return anchor


def get_topK_box(cls_score, pred_regression, anchors, topk=2):
    reg_offset = pred_regression.cpu().detach().numpy()
    scores, index = t.topk(cls_score, topk, dim=0)
    index = index.view(-1).cpu().detach().numpy()  # debug时候看下数据转换
    topk_offset = reg_offset[index, :]
    anchors = anchors[index, :]
    pred_box = box_transform_use_reg_offset(anchors, topk_offset)
    return pred_box


def norm_to_255(feature):
    max_ori = np.max(feature)
    min_ori = np.min(feature)
    feature = 0 + (255 - 0) / (max_ori - min_ori) * (feature - min_ori)
    feature = feature.astype('uint8')
    return feature


def add_box_img(img, boxes, color=(0, 255, 0)):
    """
    :param img:
    :param boxes:  cx,cy,w,h  这里的cx，cy是否是相对中心点的相对位置？
    :param color:
    """
    if boxes.ndim == 1:
        boxes = boxes[None, :]
    img = img.copy()
    img_cx = (img.shape[1] - 1) / 2
    img_cy = (img.shape[0] - 1) / 2
    for box in boxes:
        # 换成以左上角为原点的坐标
        left_top_corner = [img_cx + box[0] - box[2] / 2 + 0.5, img_cy + box[1] - box[3] / 2 + 0.5]
        right_bottom_corner = [img_cx + box[0] + box[2] / 2 - 0.5, img_cy + box[1] + box[3] / 2 - 0.5]
        left_top_corner[0] = np.clip(left_top_corner[0], 0, img.shape[1])
        right_bottom_corner[0] = np.clip(right_bottom_corner[0], 0, img.shape[1])
        left_top_corner[1] = np.clip(left_top_corner[1], 0, img.shape[0])
        right_bottom_corner[1] = np.clip(right_bottom_corner[1], 0, img.shape[0])
        img = cv2.rectangle(img, (int(left_top_corner[0]), int(left_top_corner[1])),
                            (int(right_bottom_corner[0]), int(right_bottom_corner[1])),
                            color, 2)
    return img


def use_others_model(model):
    #model_ = model['model']
    #model_ = {k.replace('featureExtract', 'sharedFeatExtra'): v for k, v in model_.items()}
    #model_ = {k.replace('conv_r1', 'conv_reg1'): v for k, v in model_.items()}
    #model_ = {k.replace('conv_r2', 'conv_reg2'): v for k, v in model_.items()}
    #model['model'] = model_
    model_ = model.copy()
    model_ = {k.replace('feature', 'sharedFeatExtra'): v for k, v in model_.items()}
    model_ = {k.replace('conv_reg_z', 'conv_reg1'): v for k, v in model_.items()}
    model_ = {k.replace('conv_reg_x', 'conv_reg2'): v for k, v in model_.items()}
    model_ = {k.replace('conv_cls_z', 'conv_cls1'): v for k, v in model_.items()}
    model_ = {k.replace('conv_cls_x', 'conv_cls2'): v for k, v in model_.items()}
    model_ = {k.replace('adjust_reg', 'regress_adjust'): v for k, v in model_.items()}
    model = model_
    return model


def get_wh_from_img_path(img_whole_path_name):
    img_gt_w, img_gt_h, img_img_w, img_img_h = float(
        img_whole_path_name.split('/')[-1].split('_')[2]), \
                                               float(img_whole_path_name.split('/')[
                                                         -1].split('_')[4][1:]), \
                                               float(img_whole_path_name.split('/')[
                                                         -1].split('_')[7]), \
                                               float(img_whole_path_name.split('/')[
                                                         -1].split('_')[-1][:-4])
    return img_gt_w, img_gt_h, img_img_w, img_img_h


# def choose_inst_img_through_exm_img(exemplar_index, trk_frames):
#     frame_range = Config.frame_range
#     if Config.update_template:
#         low_idx = max(0, exemplar_index - frame_range)
#         high_idx = min(len(trk_frames), exemplar_index + frame_range + Config.his_window + 2)
#     else:
#         low_idx = max(0, exemplar_index - frame_range)
#         high_idx = min(len(trk_frames), exemplar_index + frame_range)
#     # 在low_idx 和 high_idx里去选择一个来作为instance img， 但是为了保证每一个都能平等的选择到，
#     # 这里加入一个采样权重来达到这样的目的。
#     weights = sample_weights(exemplar_index, low_idx, high_idx, Config.sample_type)
#     instance_file_name = np.random.choice(trk_frames[low_idx:exemplar_index]
#                                           + trk_frames[exemplar_index + 1:high_idx], p=weights)
#     return instance_file_name
def choose_inst_img_through_exm_img(exemplar_index, trk_frames):
    frame_range = Config.frame_range
    low_idx = max(0, exemplar_index - frame_range)
    high_idx = min(len(trk_frames), exemplar_index + frame_range + 1)
    # 在low_idx 和 high_idx里去选择一个来作为instance img， 但是为了保证每一个都能平等的选择到，
    # 这里加入一个采样权重来达到这样的目的。
    weights = sample_weights(exemplar_index, low_idx, high_idx, Config.sample_type)
    if Config.update_template:
        # print((list(range(low_idx, exemplar_index)) + list(range(exemplar_index + 1, high_idx))).__len__())
        start_index = np.random.choice(
            (list(range(low_idx, exemplar_index)) + list(range(exemplar_index + 1, high_idx)))[0:(-Config.his_window - 1)],
            p=weights)
        instance_index = list(range(start_index, start_index + Config.his_window + 1))
    else:
        instance_index = np.random.choice(
            (list(range(low_idx, exemplar_index)) + list(range(exemplar_index + 1, high_idx))),
            p=weights)

    return instance_index


def sample_weights(center, low_idx, high_idx, sample_type='uniform'):
    """
     采样权重
    """
    if Config.update_template:
        weights = list(range(low_idx, high_idx))
        weights.remove(center)
        weights = weights[0:(-Config.his_window - 1)]
    else:
        weights = list(range(low_idx, high_idx))
        weights.remove(center)
    weights = np.array(weights)
    if sample_type == 'linear':
        weights = abs(weights - center)
    if sample_type == 'sqrt':
        weights = np.sqrt(abs(weights - center))
    if sample_type == 'uniform':
        weights = np.ones_like(weights)
    return weights / sum(weights)


def normalization(data):
    """
    :param data:
    :return: 归一化到（0,1）的数据
    """
    minVals = data.min().item()
    maxVals = data.max().item()
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    normData = data - minVals
    normData = normData / ranges
    return normData
