# model settings
model = dict(
    type='SiameseRCNN',
    pretrained='open-mmlab://resnext101_32x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=[dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        dict(
            type='BFP',
            in_channels=256,
            num_levels=5,
            refine_level=2,
            refine_type='conv',
            output_single_lvl=True),
        ],
    rpn_head=dict(
        type='RPNHead',
        in_channels=256*5,
        feat_channels=256,
        anchor_scales=[2, 4, 8, 16, 32],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[16],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    siameserpn_head=dict(
        type='SiameseRPNHead',
        in_channels=256*5,
        out_channels=256,
        anchor_scales=[2, 4, 8, 16, 32, 64],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[16],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='PSRoIPoolAfterPointwiseConv', in_channels=256*5, out_channels=10*7*7, out_size=7),
        out_channels=10,
        featmap_strides=[16]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=1,
        in_channels=10,
        fc_out_channels=4096,
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
            pos_iou_thr=0.7,
            neg_iou_thr=0.4,
            min_pos_iou=0.7,
            ignore_iof_thr=-1),
        sampler_track=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=1.0,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
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
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=8000,
        nms_post=8000,
        max_num=8000,
        nms_thr=0.9,
        min_bbox_size=0),
    siameserpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=10,
        nms_post=10,
        max_num=10,
        nms_thr=0.7,
        min_bbox_size=0
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
        nms_pre=10,
        nms_post=1,
        max_num=1,
        nms_thr=0.7,
        min_bbox_size=0
    ),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'ImageNetVIDPairDataset'
data_root = 'data/imagenet/Data/VID/train/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=5,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '../../../ImageSets/VID_train_pair_experiment.json',
        img_prefix=data_root,
        img_scale=(1024, 640),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=False,
        with_label=True,
        with_trackid=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '../../../ImageSets/VID_train_pair_experiment_1.json',
        img_prefix=data_root,
        img_scale=(1024, 640),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        with_trackid=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
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
