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
    # 还没缩放尺寸前的exemplar patch size
    size_original_exemplar = np.sqrt(w_context * h_context)
    scale_ratio = size_z / size_original_exemplar
    size_original_instance = size_original_exemplar * (size_x / size_z)
    instance_img, scale_ratio_x = crop_and_pad(img, cx, cy, size_x, size_original_instance, img_mean)
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
    if not np.array_equal(model_size, original_exemplar_size):
        img_patch = cv2.resize(img_patch_original, (model_size, model_size))
    else:
        img_patch = img_patch_original
    scale_ratio = model_size / img_patch_original.shape[0]
    return img_patch, scale_ratio


def round_up(value):
    # 保证两位小数的精确四舍五入
    return round(value + 1e-6 + 1000) - 1000


def compute_iou(anchors, gt):
    pass


def ajust_learning_rate(optimizer, decay=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']

def generate_anchors():
    pass