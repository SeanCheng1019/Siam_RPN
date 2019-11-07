import numpy as np
import cv2


def get_center(x):
    return (x - 1.0) / 2


def xyxy2cxcywh(bbox):
    return get_center(bbox[0] + bbox[2]), \
           get_center(bbox[1] + bbox[3]), \
           (bbox[2] - bbox[0]), \
           (bbox[3] - bbox[1])


def cxcywh2xyxy(bbox):
    cx, cy, w, h = bbox
    return cx - 1 / 2 * w, cy - 1 / 2 * h, cx + 1 / 2 * w, cy + 1 / 2 * h


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
    if np.array(anchors).ndim == 1:
        anchors = np.array(anchors)[None, :]
    elif np.array(box).ndim == 1:
        box = np.array(box)[None, :]
    else:
        



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
    xx, yy = np.tile(xx, (anchor_num, 1)).flatten(), np.tile(yy, (anchor_num, 1))
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor
