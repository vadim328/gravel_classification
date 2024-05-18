affine_scale = 0.9
albu_train_transforms = [
    dict(p=0.5, type='HorizontalFlip'),
    dict(
        height=768,
        ratio=(
            0.8,
            1.2,
        ),
        scale=(
            0.8,
            1.2,
        ),
        type='RandomResizedCrop',
        width=1280),
    dict(p=0.25, type='RandomRotate90'),
    dict(
        interpolation=1,
        p=0.5,
        rotate_limit=15,
        scale_limit=0.1,
        shift_limit=0.1,
        type='ShiftScaleRotate'),
    dict(max_h_size=32, max_w_size=32, num_holes=12, p=0.05, type='Cutout'),
    dict(clip_limit=2, p=0.1, type='CLAHE'),
    dict(p=0.2, type='RandomBrightnessContrast'),
    dict(p=0.2, type='GaussNoise'),
    dict(blur_limit=3, p=0.2, type='MotionBlur'),
    dict(p=0.2, type='ISONoise'),
    dict(p=0.25, quality_lower=15, quality_upper=30, type='ImageCompression'),
]
classes = (
    '0-20',
    '0-5',
    '20-25',
    '20-40',
    '25-60',
    '40-70',
    '5-20',
)
close_mosaic_epochs = 12
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        priority=49,
        strict_load=False,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=188,
        switch_pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                transforms=[
                    dict(p=0.5, type='HorizontalFlip'),
                    dict(
                        height=768,
                        ratio=(
                            0.8,
                            1.2,
                        ),
                        scale=(
                            0.8,
                            1.2,
                        ),
                        type='RandomResizedCrop',
                        width=1280),
                    dict(p=0.25, type='RandomRotate90'),
                    dict(
                        interpolation=1,
                        p=0.5,
                        rotate_limit=15,
                        scale_limit=0.1,
                        shift_limit=0.1,
                        type='ShiftScaleRotate'),
                    dict(
                        max_h_size=32,
                        max_w_size=32,
                        num_holes=12,
                        p=0.05,
                        type='Cutout'),
                    dict(clip_limit=2, p=0.1, type='CLAHE'),
                    dict(p=0.2, type='RandomBrightnessContrast'),
                    dict(p=0.2, type='GaussNoise'),
                    dict(blur_limit=3, p=0.2, type='MotionBlur'),
                    dict(p=0.2, type='ISONoise'),
                    dict(
                        p=0.25,
                        quality_lower=15,
                        quality_upper=30,
                        type='ImageCompression'),
                ],
                type='mmdet.Albu'),
            dict(scale=(
                1280,
                768,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=True,
                pad_val=dict(img=114.0),
                scale=(
                    1280,
                    768,
                ),
                type='LetterResize'),
            dict(
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=10,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(
                    0.09999999999999998,
                    1.9,
                ),
                type='YOLOv5RandomAffine'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'flip',
                    'flip_direction',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='mmdet.PipelineSwitchHook'),
]
data_root = '/openmmlab/mmyolo/external_data/datasets/dataset_detection_and_classification/'
deepen_factor = 0.67
default_hooks = dict(
    checkpoint=dict(
        interval=1, max_keep_ckpts=2, save_best='auto', type='CheckpointHook'),
    logger=dict(interval=25, type='LoggerHook'),
    param_scheduler=dict(
        lr_factor=0.01,
        max_epochs=200,
        scheduler_type='linear',
        type='YOLOv5ParamSchedulerHook',
        warmup_mim_iter=500),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmyolo'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scale = (
    1280,
    768,
)
last_stage_out_channels = 768
last_transform = [
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
]
launcher = 'none'
load_from = '/openmmlab/mmyolo/external_data/models/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
loss_bbox_weight = 7.5
loss_cls_weight = 0.5
loss_dfl_weight = 0.375
lr_factor = 0.01
max_aspect_ratio = 10
max_epochs = 200
max_keep_ckpts = 2
mixup_prob = 0.1
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='P5',
        deepen_factor=0.67,
        last_stage_out_channels=768,
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        type='YOLOv8CSPDarknet',
        widen_factor=0.75),
    bbox_head=dict(
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        head_module=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            featmap_strides=[
                8,
                16,
                32,
            ],
            in_channels=[
                256,
                512,
                768,
            ],
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            num_classes=7,
            reg_max=16,
            type='YOLOv8HeadModule',
            widen_factor=0.75),
        loss_bbox=dict(
            bbox_format='xyxy',
            iou_mode='ciou',
            loss_weight=7.5,
            reduction='sum',
            return_iou=False,
            type='IoULoss'),
        loss_cls=dict(
            loss_weight=0.5,
            reduction='none',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        loss_dfl=dict(
            loss_weight=0.375,
            reduction='mean',
            type='mmdet.DistributionFocalLoss'),
        prior_generator=dict(
            offset=0.5, strides=[
                8,
                16,
                32,
            ], type='mmdet.MlvlPointGenerator'),
        type='YOLOv8Head'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='YOLOv5DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        deepen_factor=0.67,
        in_channels=[
            256,
            512,
            768,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_csp_blocks=3,
        out_channels=[
            256,
            512,
            768,
        ],
        type='YOLOv8PAFPN',
        widen_factor=0.75),
    test_cfg=dict(
        max_per_img=100,
        multi_label=False,
        nms=dict(iou_threshold=0.7, type='nms'),
        nms_pre=10000,
        score_thr=0.01),
    train_cfg=dict(
        assigner=dict(
            alpha=0.5,
            beta=6.0,
            eps=1e-09,
            num_classes=7,
            topk=10,
            type='BatchTaskAlignedAssigner',
            use_ciou=True)),
    type='YOLODetector')
mosaic_affine_transform = [
    dict(
        img_scale=(
            1280,
            768,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                transforms=[
                    dict(p=0.5, type='HorizontalFlip'),
                    dict(
                        height=768,
                        ratio=(
                            0.8,
                            1.2,
                        ),
                        scale=(
                            0.8,
                            1.2,
                        ),
                        type='RandomResizedCrop',
                        width=1280),
                    dict(p=0.25, type='RandomRotate90'),
                    dict(
                        interpolation=1,
                        p=0.5,
                        rotate_limit=15,
                        scale_limit=0.1,
                        shift_limit=0.1,
                        type='ShiftScaleRotate'),
                    dict(
                        max_h_size=32,
                        max_w_size=32,
                        num_holes=12,
                        p=0.05,
                        type='Cutout'),
                    dict(clip_limit=2, p=0.1, type='CLAHE'),
                    dict(p=0.2, type='RandomBrightnessContrast'),
                    dict(p=0.2, type='GaussNoise'),
                    dict(blur_limit=3, p=0.2, type='MotionBlur'),
                    dict(p=0.2, type='ISONoise'),
                    dict(
                        p=0.25,
                        quality_lower=15,
                        quality_upper=30,
                        type='ImageCompression'),
                ],
                type='mmdet.Albu'),
        ],
        prob=1.0,
        type='Mosaic'),
    dict(
        border=(
            -640,
            -384,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=10,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.09999999999999998,
            1.9,
        ),
        type='YOLOv5RandomAffine'),
]
norm_cfg = dict(eps=0.001, momentum=0.03, type='BN')
num_classes = 7
optim_wrapper = dict(
    clip_grad=dict(max_norm=10.0),
    constructor='YOLOv5OptimizerConstructor',
    optimizer=dict(
        batch_size_per_gpu=8,
        lr=0.01,
        momentum=0.937,
        nesterov=True,
        type='SGD',
        weight_decay=0.0005),
    type='OptimWrapper')
pre_transform = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.5, type='HorizontalFlip'),
            dict(
                height=768,
                ratio=(
                    0.8,
                    1.2,
                ),
                scale=(
                    0.8,
                    1.2,
                ),
                type='RandomResizedCrop',
                width=1280),
            dict(p=0.25, type='RandomRotate90'),
            dict(
                interpolation=1,
                p=0.5,
                rotate_limit=15,
                scale_limit=0.1,
                shift_limit=0.1,
                type='ShiftScaleRotate'),
            dict(
                max_h_size=32,
                max_w_size=32,
                num_holes=12,
                p=0.05,
                type='Cutout'),
            dict(clip_limit=2, p=0.1, type='CLAHE'),
            dict(p=0.2, type='RandomBrightnessContrast'),
            dict(p=0.2, type='GaussNoise'),
            dict(blur_limit=3, p=0.2, type='MotionBlur'),
            dict(p=0.2, type='ISONoise'),
            dict(
                p=0.25,
                quality_lower=15,
                quality_upper=30,
                type='ImageCompression'),
        ],
        type='mmdet.Albu'),
]
resume = False
save_epoch_intervals = 1
strides = [
    8,
    16,
    32,
]
tal_alpha = 0.5
tal_beta = 6.0
tal_topk = 10
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotation_test.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img='images/'),
        data_root=
        '/openmmlab/mmyolo/external_data/datasets/dataset_detection_and_classification/test/',
        metainfo=dict(
            classes=(
                '0-20',
                '0-5',
                '20-25',
                '20-40',
                '25-60',
                '40-70',
                '5-20',
            )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                1280,
                768,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    1280,
                    768,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/openmmlab/mmyolo/external_data/datasets/dataset_detection_and_classification//test/annotation_test.json',
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(scale=(
        1280,
        768,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=False,
        pad_val=dict(img=114),
        scale=(
            1280,
            768,
        ),
        type='LetterResize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
        ),
        type='mmdet.PackDetInputs'),
]
train_cfg = dict(
    dynamic_intervals=[
        (
            188,
            1,
        ),
    ],
    max_epochs=200,
    type='EpochBasedTrainLoop',
    val_interval=1)
train_dataloader = dict(
    batch_size=12,
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        datasets=[
            dict(
                ann_file='annotation_train.json',
                data_prefix=dict(img='images/'),
                data_root=
                '/openmmlab/mmyolo/external_data/datasets/dataset_detection_and_classification/train/',
                filter_cfg=dict(filter_empty_gt=False, min_size=32),
                metainfo=dict(
                    classes=(
                        '0-20',
                        '0-5',
                        '20-25',
                        '20-40',
                        '25-60',
                        '40-70',
                        '5-20',
                    )),
                pipeline=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        bbox_params=dict(
                            format='pascal_voc',
                            label_fields=[
                                'gt_bboxes_labels',
                                'gt_ignore_flags',
                            ],
                            type='BboxParams'),
                        keymap=dict(gt_bboxes='bboxes', img='image'),
                        transforms=[
                            dict(p=0.5, type='HorizontalFlip'),
                            dict(
                                height=768,
                                ratio=(
                                    0.8,
                                    1.2,
                                ),
                                scale=(
                                    0.8,
                                    1.2,
                                ),
                                type='RandomResizedCrop',
                                width=1280),
                            dict(p=0.25, type='RandomRotate90'),
                            dict(
                                interpolation=1,
                                p=0.5,
                                rotate_limit=15,
                                scale_limit=0.1,
                                shift_limit=0.1,
                                type='ShiftScaleRotate'),
                            dict(
                                max_h_size=32,
                                max_w_size=32,
                                num_holes=12,
                                p=0.05,
                                type='Cutout'),
                            dict(clip_limit=2, p=0.1, type='CLAHE'),
                            dict(p=0.2, type='RandomBrightnessContrast'),
                            dict(p=0.2, type='GaussNoise'),
                            dict(blur_limit=3, p=0.2, type='MotionBlur'),
                            dict(p=0.2, type='ISONoise'),
                            dict(
                                p=0.25,
                                quality_lower=15,
                                quality_upper=30,
                                type='ImageCompression'),
                        ],
                        type='mmdet.Albu'),
                    dict(
                        img_scale=(
                            1280,
                            768,
                        ),
                        pad_val=114.0,
                        pre_transform=[
                            dict(backend_args=None, type='LoadImageFromFile'),
                            dict(type='LoadAnnotations', with_bbox=True),
                            dict(
                                bbox_params=dict(
                                    format='pascal_voc',
                                    label_fields=[
                                        'gt_bboxes_labels',
                                        'gt_ignore_flags',
                                    ],
                                    type='BboxParams'),
                                keymap=dict(gt_bboxes='bboxes', img='image'),
                                transforms=[
                                    dict(p=0.5, type='HorizontalFlip'),
                                    dict(
                                        height=768,
                                        ratio=(
                                            0.8,
                                            1.2,
                                        ),
                                        scale=(
                                            0.8,
                                            1.2,
                                        ),
                                        type='RandomResizedCrop',
                                        width=1280),
                                    dict(p=0.25, type='RandomRotate90'),
                                    dict(
                                        interpolation=1,
                                        p=0.5,
                                        rotate_limit=15,
                                        scale_limit=0.1,
                                        shift_limit=0.1,
                                        type='ShiftScaleRotate'),
                                    dict(
                                        max_h_size=32,
                                        max_w_size=32,
                                        num_holes=12,
                                        p=0.05,
                                        type='Cutout'),
                                    dict(clip_limit=2, p=0.1, type='CLAHE'),
                                    dict(
                                        p=0.2,
                                        type='RandomBrightnessContrast'),
                                    dict(p=0.2, type='GaussNoise'),
                                    dict(
                                        blur_limit=3, p=0.2,
                                        type='MotionBlur'),
                                    dict(p=0.2, type='ISONoise'),
                                    dict(
                                        p=0.25,
                                        quality_lower=15,
                                        quality_upper=30,
                                        type='ImageCompression'),
                                ],
                                type='mmdet.Albu'),
                        ],
                        prob=1.0,
                        type='Mosaic'),
                    dict(
                        border=(
                            -640,
                            -384,
                        ),
                        border_val=(
                            114,
                            114,
                            114,
                        ),
                        max_aspect_ratio=10,
                        max_rotate_degree=0.0,
                        max_shear_degree=0.0,
                        scaling_ratio_range=(
                            0.09999999999999998,
                            1.9,
                        ),
                        type='YOLOv5RandomAffine'),
                    dict(
                        pre_transform=[
                            dict(backend_args=None, type='LoadImageFromFile'),
                            dict(type='LoadAnnotations', with_bbox=True),
                            dict(
                                bbox_params=dict(
                                    format='pascal_voc',
                                    label_fields=[
                                        'gt_bboxes_labels',
                                        'gt_ignore_flags',
                                    ],
                                    type='BboxParams'),
                                keymap=dict(gt_bboxes='bboxes', img='image'),
                                transforms=[
                                    dict(p=0.5, type='HorizontalFlip'),
                                    dict(
                                        height=768,
                                        ratio=(
                                            0.8,
                                            1.2,
                                        ),
                                        scale=(
                                            0.8,
                                            1.2,
                                        ),
                                        type='RandomResizedCrop',
                                        width=1280),
                                    dict(p=0.25, type='RandomRotate90'),
                                    dict(
                                        interpolation=1,
                                        p=0.5,
                                        rotate_limit=15,
                                        scale_limit=0.1,
                                        shift_limit=0.1,
                                        type='ShiftScaleRotate'),
                                    dict(
                                        max_h_size=32,
                                        max_w_size=32,
                                        num_holes=12,
                                        p=0.05,
                                        type='Cutout'),
                                    dict(clip_limit=2, p=0.1, type='CLAHE'),
                                    dict(
                                        p=0.2,
                                        type='RandomBrightnessContrast'),
                                    dict(p=0.2, type='GaussNoise'),
                                    dict(
                                        blur_limit=3, p=0.2,
                                        type='MotionBlur'),
                                    dict(p=0.2, type='ISONoise'),
                                    dict(
                                        p=0.25,
                                        quality_lower=15,
                                        quality_upper=30,
                                        type='ImageCompression'),
                                ],
                                type='mmdet.Albu'),
                            dict(
                                img_scale=(
                                    1280,
                                    768,
                                ),
                                pad_val=114.0,
                                pre_transform=[
                                    dict(
                                        backend_args=None,
                                        type='LoadImageFromFile'),
                                    dict(
                                        type='LoadAnnotations',
                                        with_bbox=True),
                                    dict(
                                        bbox_params=dict(
                                            format='pascal_voc',
                                            label_fields=[
                                                'gt_bboxes_labels',
                                                'gt_ignore_flags',
                                            ],
                                            type='BboxParams'),
                                        keymap=dict(
                                            gt_bboxes='bboxes', img='image'),
                                        transforms=[
                                            dict(p=0.5, type='HorizontalFlip'),
                                            dict(
                                                height=768,
                                                ratio=(
                                                    0.8,
                                                    1.2,
                                                ),
                                                scale=(
                                                    0.8,
                                                    1.2,
                                                ),
                                                type='RandomResizedCrop',
                                                width=1280),
                                            dict(
                                                p=0.25, type='RandomRotate90'),
                                            dict(
                                                interpolation=1,
                                                p=0.5,
                                                rotate_limit=15,
                                                scale_limit=0.1,
                                                shift_limit=0.1,
                                                type='ShiftScaleRotate'),
                                            dict(
                                                max_h_size=32,
                                                max_w_size=32,
                                                num_holes=12,
                                                p=0.05,
                                                type='Cutout'),
                                            dict(
                                                clip_limit=2,
                                                p=0.1,
                                                type='CLAHE'),
                                            dict(
                                                p=0.2,
                                                type='RandomBrightnessContrast'
                                            ),
                                            dict(p=0.2, type='GaussNoise'),
                                            dict(
                                                blur_limit=3,
                                                p=0.2,
                                                type='MotionBlur'),
                                            dict(p=0.2, type='ISONoise'),
                                            dict(
                                                p=0.25,
                                                quality_lower=15,
                                                quality_upper=30,
                                                type='ImageCompression'),
                                        ],
                                        type='mmdet.Albu'),
                                ],
                                prob=1.0,
                                type='Mosaic'),
                            dict(
                                border=(
                                    -640,
                                    -384,
                                ),
                                border_val=(
                                    114,
                                    114,
                                    114,
                                ),
                                max_aspect_ratio=10,
                                max_rotate_degree=0.0,
                                max_shear_degree=0.0,
                                scaling_ratio_range=(
                                    0.09999999999999998,
                                    1.9,
                                ),
                                type='YOLOv5RandomAffine'),
                        ],
                        prob=0.1,
                        type='YOLOv5MixUp'),
                    dict(prob=0.5, type='mmdet.RandomFlip'),
                    dict(
                        meta_keys=(
                            'img_id',
                            'img_path',
                            'ori_shape',
                            'img_shape',
                            'flip',
                            'flip_direction',
                        ),
                        type='mmdet.PackDetInputs'),
                ],
                type='YOLOv5CocoDataset'),
        ],
        ignore_keys=[
            'dataset_type',
        ],
        type='ConcatDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.5, type='HorizontalFlip'),
            dict(
                height=768,
                ratio=(
                    0.8,
                    1.2,
                ),
                scale=(
                    0.8,
                    1.2,
                ),
                type='RandomResizedCrop',
                width=1280),
            dict(p=0.25, type='RandomRotate90'),
            dict(
                interpolation=1,
                p=0.5,
                rotate_limit=15,
                scale_limit=0.1,
                shift_limit=0.1,
                type='ShiftScaleRotate'),
            dict(
                max_h_size=32,
                max_w_size=32,
                num_holes=12,
                p=0.05,
                type='Cutout'),
            dict(clip_limit=2, p=0.1, type='CLAHE'),
            dict(p=0.2, type='RandomBrightnessContrast'),
            dict(p=0.2, type='GaussNoise'),
            dict(blur_limit=3, p=0.2, type='MotionBlur'),
            dict(p=0.2, type='ISONoise'),
            dict(
                p=0.25,
                quality_lower=15,
                quality_upper=30,
                type='ImageCompression'),
        ],
        type='mmdet.Albu'),
    dict(
        img_scale=(
            1280,
            768,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                transforms=[
                    dict(p=0.5, type='HorizontalFlip'),
                    dict(
                        height=768,
                        ratio=(
                            0.8,
                            1.2,
                        ),
                        scale=(
                            0.8,
                            1.2,
                        ),
                        type='RandomResizedCrop',
                        width=1280),
                    dict(p=0.25, type='RandomRotate90'),
                    dict(
                        interpolation=1,
                        p=0.5,
                        rotate_limit=15,
                        scale_limit=0.1,
                        shift_limit=0.1,
                        type='ShiftScaleRotate'),
                    dict(
                        max_h_size=32,
                        max_w_size=32,
                        num_holes=12,
                        p=0.05,
                        type='Cutout'),
                    dict(clip_limit=2, p=0.1, type='CLAHE'),
                    dict(p=0.2, type='RandomBrightnessContrast'),
                    dict(p=0.2, type='GaussNoise'),
                    dict(blur_limit=3, p=0.2, type='MotionBlur'),
                    dict(p=0.2, type='ISONoise'),
                    dict(
                        p=0.25,
                        quality_lower=15,
                        quality_upper=30,
                        type='ImageCompression'),
                ],
                type='mmdet.Albu'),
        ],
        prob=1.0,
        type='Mosaic'),
    dict(
        border=(
            -640,
            -384,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=10,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.09999999999999998,
            1.9,
        ),
        type='YOLOv5RandomAffine'),
    dict(
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                transforms=[
                    dict(p=0.5, type='HorizontalFlip'),
                    dict(
                        height=768,
                        ratio=(
                            0.8,
                            1.2,
                        ),
                        scale=(
                            0.8,
                            1.2,
                        ),
                        type='RandomResizedCrop',
                        width=1280),
                    dict(p=0.25, type='RandomRotate90'),
                    dict(
                        interpolation=1,
                        p=0.5,
                        rotate_limit=15,
                        scale_limit=0.1,
                        shift_limit=0.1,
                        type='ShiftScaleRotate'),
                    dict(
                        max_h_size=32,
                        max_w_size=32,
                        num_holes=12,
                        p=0.05,
                        type='Cutout'),
                    dict(clip_limit=2, p=0.1, type='CLAHE'),
                    dict(p=0.2, type='RandomBrightnessContrast'),
                    dict(p=0.2, type='GaussNoise'),
                    dict(blur_limit=3, p=0.2, type='MotionBlur'),
                    dict(p=0.2, type='ISONoise'),
                    dict(
                        p=0.25,
                        quality_lower=15,
                        quality_upper=30,
                        type='ImageCompression'),
                ],
                type='mmdet.Albu'),
            dict(
                img_scale=(
                    1280,
                    768,
                ),
                pad_val=114.0,
                pre_transform=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        bbox_params=dict(
                            format='pascal_voc',
                            label_fields=[
                                'gt_bboxes_labels',
                                'gt_ignore_flags',
                            ],
                            type='BboxParams'),
                        keymap=dict(gt_bboxes='bboxes', img='image'),
                        transforms=[
                            dict(p=0.5, type='HorizontalFlip'),
                            dict(
                                height=768,
                                ratio=(
                                    0.8,
                                    1.2,
                                ),
                                scale=(
                                    0.8,
                                    1.2,
                                ),
                                type='RandomResizedCrop',
                                width=1280),
                            dict(p=0.25, type='RandomRotate90'),
                            dict(
                                interpolation=1,
                                p=0.5,
                                rotate_limit=15,
                                scale_limit=0.1,
                                shift_limit=0.1,
                                type='ShiftScaleRotate'),
                            dict(
                                max_h_size=32,
                                max_w_size=32,
                                num_holes=12,
                                p=0.05,
                                type='Cutout'),
                            dict(clip_limit=2, p=0.1, type='CLAHE'),
                            dict(p=0.2, type='RandomBrightnessContrast'),
                            dict(p=0.2, type='GaussNoise'),
                            dict(blur_limit=3, p=0.2, type='MotionBlur'),
                            dict(p=0.2, type='ISONoise'),
                            dict(
                                p=0.25,
                                quality_lower=15,
                                quality_upper=30,
                                type='ImageCompression'),
                        ],
                        type='mmdet.Albu'),
                ],
                prob=1.0,
                type='Mosaic'),
            dict(
                border=(
                    -640,
                    -384,
                ),
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=10,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(
                    0.09999999999999998,
                    1.9,
                ),
                type='YOLOv5RandomAffine'),
        ],
        prob=0.1,
        type='YOLOv5MixUp'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.5, type='HorizontalFlip'),
            dict(
                height=768,
                ratio=(
                    0.8,
                    1.2,
                ),
                scale=(
                    0.8,
                    1.2,
                ),
                type='RandomResizedCrop',
                width=1280),
            dict(p=0.25, type='RandomRotate90'),
            dict(
                interpolation=1,
                p=0.5,
                rotate_limit=15,
                scale_limit=0.1,
                shift_limit=0.1,
                type='ShiftScaleRotate'),
            dict(
                max_h_size=32,
                max_w_size=32,
                num_holes=12,
                p=0.05,
                type='Cutout'),
            dict(clip_limit=2, p=0.1, type='CLAHE'),
            dict(p=0.2, type='RandomBrightnessContrast'),
            dict(p=0.2, type='GaussNoise'),
            dict(blur_limit=3, p=0.2, type='MotionBlur'),
            dict(p=0.2, type='ISONoise'),
            dict(
                p=0.25,
                quality_lower=15,
                quality_upper=30,
                type='ImageCompression'),
        ],
        type='mmdet.Albu'),
    dict(scale=(
        1280,
        768,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=True,
        pad_val=dict(img=114.0),
        scale=(
            1280,
            768,
        ),
        type='LetterResize'),
    dict(
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=10,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.09999999999999998,
            1.9,
        ),
        type='YOLOv5RandomAffine'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotation_val.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img='images/'),
        data_root=
        '/openmmlab/mmyolo/external_data/datasets/dataset_detection_and_classification/val/',
        metainfo=dict(
            classes=(
                '0-20',
                '0-5',
                '20-25',
                '20-40',
                '25-60',
                '40-70',
                '5-20',
            )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                1280,
                768,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    1280,
                    768,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/openmmlab/mmyolo/external_data/datasets/dataset_detection_and_classification//val/annotation_val.json',
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
val_interval_stage2 = 1
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
weight_decay = 0.0005
widen_factor = 0.75
work_dir = './work_dirs/yolov8'
