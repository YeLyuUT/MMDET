dataset_type = 'ImageNetDETVIDDataset'
data_root = 'data/imagenet/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/VID_val.json',
        img_prefix=data_root + '/Data/VID/val',
        img_scale=[(1300, 560)],#[(1100,720),(1000, 640),(900,560)],
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
