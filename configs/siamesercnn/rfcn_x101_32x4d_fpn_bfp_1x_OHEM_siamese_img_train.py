# model settings
model = dict(
    type='SiameseRCNN',
    pretrained='open-mmlab://msra/hrnetv2_w32',
    img_train=True,
    backbone=dict(
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256)))),
    neck=dict(
            type='BFP',
            in_channels=[32, 64, 128, 256],
            num_levels=4,
            refine_level=1,
            out_channels=256,
            output_single_lvl = True),
    rpn_head=dict(
        type='RPNHead',
        in_channels=480,
        feat_channels=256,
        anchor_scales=[2, 4, 8, 16, 32, 64],
        anchor_ratios=[0.33, 0.66, 1.0, 1.5, 3.0],
        anchor_strides=[8],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    siameserpn_head=dict(
        type='SiameseRPNHead',
        in_channels=480,
        out_channels=256,
        feat_strides=[8],
        target_means=[.0, .0, .0, .0],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        loss_cls=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='PSRoIPoolAfterPointwiseConv', in_channels=480, out_channels=10*7*7, out_size=7, n_prev=3),
        out_channels=10,
        featmap_strides=[8]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=1,
        in_channels=10,
        fc_out_channels=2048,
        roi_feat_size=7,
        num_classes=31,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    siameserpn = dict(
        assigner_track=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler_track=dict(
            type='RandomSampler',
            num=128,
            pos_fraction=1.0,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=8000,
        nms_post=8000,
        max_num=8000,
        nms_thr=0.7,
        min_bbox_size=0),
    siameserpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0,
        score_threshold=0.0,
        TRACK = dict(
        PENALTY_K = 0.15,
        WINDOW_INFLUENCE = 0.0,),
        ),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='OHEMSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    siameserpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0,
        score_threshold=0.7,
        TRACK = dict(
        PENALTY_K = 0.15,
        WINDOW_INFLUENCE = 0.0,
        ),
    ),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'ImageNetDETVIDDataset'
data_root = 'data/imagenet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/ImageSets/DET_train_30classes_experiment.json',
        img_prefix=data_root,
        multiscale_mode='range',
        img_scale=[(1333, 800), (800, 480)],
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=False,
        with_label=True,),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/ImageSets/DET_train_30classes_experiment.json',
        img_prefix=data_root,
        img_scale=[(1000, 640)],#[(1100,720),(1000, 640),(900,560)],
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=1.0 / 3,
    step=[500])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        #dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 1
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/siamese_rcnn_experiment'
load_from = None
resume_from = None
workflow = [('train', 1)]
