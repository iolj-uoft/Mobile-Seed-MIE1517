_base_ = [
    '../_base_/models/Mobile_Seed.py',
    '../_base_/default_runtime.py',
]

dataset_type = 'CustomDataset'
data_root = 'data/dash_cam_processed/'
classes = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider',
    'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]
palette = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
    [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
]

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='LoadEdgeFromFile', edge_dir='data/dash_cam_processed/edge_dir/train', edge_suffix='_edge.png'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'gt_semantic_sebound']),
]

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        pipeline=train_pipeline,
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
    )
)

model = dict(
    pretrained='./ckpt/GCA.pth',  # from Mobile-Seed
    backbone=dict(type='AFFormer_for_MS_tiny'),
    decode_head=[
        dict(
            type="BoundaryHead",
            bound_channels=[16, 16, 32, 32],
            bound_ratio=2,
            in_channels=[16, 64, 216, 216],
            in_index=[1, 3, 5, 6],
            channels=96,
            num_classes=1,
            loss_decode=dict(
                type='ML_BCELoss', use_sigmoid=True, loss_weight=1.0, loss_name="loss_be"
            )
        ),
        dict(
            type="RefineHead",
            fuse_channel=96,
            in_channels=[216],
            in_index=[-1],
            channels=256,
            num_classes=19,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, loss_name="loss_ce"
            )
        ),
    ]
)

optimizer = dict(
    type='AdamW',
    lr=0.0004 * 3 / 8,
    betas=(0.9, 0.999),
    weight_decay=0.01
)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)

runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(interval=4000)
evaluation = dict(interval=4000, metric='mIoU')
find_unused_parameters = True
