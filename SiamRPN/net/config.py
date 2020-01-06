import numpy as np


class Config:
    instance_size = 271
    exemplar_size = 127
    instance_crop_size = 511
    context_margin_amount = 0.5
    total_stride = 8
    valid_map_size = int((instance_size - exemplar_size) / total_stride) + 1
    pairs_per_sequence_per_epoch = 2
    anchor_scales = np.array([8, ])
    anchor_ratio = np.array([0.33, 0.5, 1, 2, 3])
    scale_range = (0.001, 0.7)  # what's this
    anchor_num = len(anchor_ratio) * len(anchor_scales)
    anchor_base_size = 8
    score_map_size = (instance_size - exemplar_size) // total_stride + 1
    scale_range = (0.001, 0.7)
    ratio_range = (0.1, 10)
    frame_range = 100
    sample_type = 'uniform'
    gray_ratio = 0.25
    scale_resize = 0.15  # 训练时对instance_img的缩放
    max_shift = 12  # 训练时对中心的最大偏移量
    exemplar_stretch = False
    iou_pos_threshold = 0.6
    iou_neg_threshold = 0.3
    train_batch_size = 32
    valid_batch_size = 8
    train_num_workers = 0
    valid_num_workers = 0
    log_dir = '../data/logs'
    train_ratio = 0.99
    seed = 6666
    start_lr = 3e-2
    end_lr = 1e-5
    epoch = 50
    lr = np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[0]  # 构造一系列lr
    gamma = np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[1] / \
            np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[0]
    momentum = 0.9
    weight_dacay = 0.0005  # weight decay of optimizator
    pretrained_model = "/media/csy/62ac73e0-814c-4dba-b59d-676690aca14b/PycharmProjects/Siam_RPN/SiamRPN/model/alexnet.pth"
    fix_former_3_layers = True
    num_pos = 16
    num_neg = 48
    ohem_pos = False
    ohem_neg = False
    ohem_reg = False
    lamb = 5
    clip = 10
    show_interval = 3
    save_interval = 1
    show_topK = 1
    penalty_k = 0.04
    window_influence = 0.44
    track_lr = 0.4
    min_scale = 0.1
    max_scale = 10
    show_net_feature = True
    use_others = False
    memAlign = False
    his_window = 5  # 5张历史帧
    stmm_train_batch_size = 8
    stmm_valid_batch_size = 1
    update_template = False
    template_combinition_coef = 0.6

