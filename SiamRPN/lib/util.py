import numpy as np
import cv2


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
    y2 = y1 + bboxes[:, 2:3] - 1
    return np.concatenate([x1, y1, x2, y2], 1), x1, x2, y1, y2


def get_instance_img(img, bbox, size_z, size_x, context_margin_amount, img_mean=None):
    cx, cy, w, h = bbox
    w_context = w + context_margin_amount * (w + h)
    h_context = h + context_margin_amount * (w + h)
    # 还没缩放尺寸前的exemplar patch size， 是个正方形patch，但是尺寸还没缩放到127/255之类的
    size_original_exemplar = np.sqrt(w_context * h_context)
    scale_ratio = size_z / size_original_exemplar
    size_original_instance = size_original_exemplar * (size_x / size_z)
    instance_img, scale_ratio_x = crop_and_pad(img, cx, cy, size_x, size_original_instance, img_mean)
    # 因为resize过，所以目标的尺寸要从原图乘上resize的ratio
    w_instance = w * scale_ratio_x
    h_instance = h * scale_ratio_x
    return instance_img, w_instance, h_instance, scale_ratio_x


def get_exemplar_img(img, bbox, size_z, size_x, context_margin_amount, img_mean=None):
    cx, cy, w, h = bbox
    w_context = w + context_margin_amount * (w + h)
    h_context = h + context_margin_amount * (w + h)
    size_original_exemplar = np.sqrt(w_context * h_context)
    scale_ration_z = size_z / size_original_exemplar
    exemplar_img, _ = crop_and_pad(img, cx, cy, size_z, size_original_exemplar, img_mean)
    return exemplar_img, scale_ration_z, size_original_exemplar


def crop_and_pad(img, cx, cy, model_size, original_exemplar_size, img_mean=None):
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
        img_patch_original = tmp_img[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
    else:
        # 无填充的情况下，直接在原图上裁剪
        img_patch_original = img[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
    if not np.array_equal(model_size, original_exemplar_size):  # resize到config里定义的尺寸。
        img_patch = cv2.resize(img_patch_original, (model_size, model_size))
    else:
        img_patch = img_patch_original
    scale_ratio = model_size / img_patch_original.shape[0]
    return img_patch, scale_ratio


def round_up(value):
    # 保证两位小数的精确四舍五入
    return round(value + 1e-6 + 1000) - 1000


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
    regression_target = np.hstack((delta_x, delta_y, delta_w, delta_h))
    return regression_target


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
    # anchor_x1 = anchors[:, :1] - anchors[:, 2:3] / 2 + 0.5
    # anchor_x2 = anchors[:, :1] + anchors[:, 2:3] / 2 - 0.5
    # anchor_y1 = anchors[:, 1:2] - anchors[:, 3:] / 2 + 0.5
    # anchor_y2 = anchors[:, 1:2] + anchors[:, 3:] / 2 - 0.5
    # gt_x1 = gt_box[:, :1] - gt_box[:, 2:3] / 2 + 0.5
    # gt_x2 = gt_box[:, :1] + gt_box[:, 2:3] / 2 - 0.5
    # gt_y1 = gt_box[:, 1:2] - gt_box[:, 3:] / 2 + 0.5
    # gt_y2 = gt_box[:, 1:2] + gt_box[:, 3:] / 2 - 0.5
    overlap_x1 = np.max([anchor_x1, gt_x1], axis=0)
    overlap_x2 = np.min([anchor_x2, gt_x2], axis=0)
    overlap_y1 = np.max([anchor_y1, gt_y1], axis=0)
    overlap_y2 = np.min([anchor_y2, gt_y2], axis=0)
    # 要注意到没有交集的情况，x2-x1就会出现负数，就没有意义，所以要限制最小是0
    overlap_area = np.max([(overlap_x2 - overlap_x1), np.zeros(overlap_x1)], axis=0) \
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
    anchor_num = len(scales) * len(ratios)  # 每个位置的锚框数量
    anchor = np.zero((anchor_num, 4), dtype=np.float32)
    size = np.square(base_size)
    count = 0
    '''
    原来的面积是 base_size*base_size=size, 改变后是base_size*(base_size*ratio)=size
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
    xx, yy = np.tile(xx, (anchor_num, 1)).flatten(), np.tile(yy, (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor
