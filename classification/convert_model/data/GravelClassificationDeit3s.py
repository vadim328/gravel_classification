# ---- DEIT3s ----

classes = ["0-20", "0-5", "20-25", "20-40", "25-60", "40-70", "5-20"]
num_classes=len(classes) 
max_epochs=100

_base_ = [
    '../_base_/default_runtime.py',
]

# ---- MODEL ----

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='DeiT3',
        arch='s',
        img_size=384,
        patch_size=16,
        drop_path_rate=0.0,
        init_cfg = dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/deit3/deit3-small-p16_in21k-pre_3rdparty_in1k-384px_20221009-de116dd7.pth',
            prefix='backbone')
    ),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=num_classes,
        in_channels=384,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]))

# ---- SCHEDULES ----

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.1),
    # specific to vit pretrain
    paramwise_cfg=dict(custom_keys={
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0)
    }),
)
# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-2,
        by_epoch=True,
        begin=0,
        end=60,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=80,
        by_epoch=True,
        begin=60,
        end=100,
    )
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
#auto_scale_lr = dict(base_batch_size=32)


# ---- DATA SETTINGS ----

# We re-organized the dataset as `CustomDataset` format.
dataset_type = 'ImageNet'
data_preprocessor = dict(
    num_classes=num_classes,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

# train, val, test setting
albu_train_transforms = [
    dict(type="HorizontalFlip", p=0.5),
    dict(type="RandomRotate90", p=0.25),
    dict(type="ShiftScaleRotate", shift_limit=0.1, scale_limit=0.1, rotate_limit=15, interpolation=1, p=0.5),
    dict(type="CLAHE", clip_limit=2, p=0.1),
    dict(type="RandomBrightnessContrast", p=0.2),
    dict(type="GaussNoise", p=0.2),
    dict(type="MotionBlur", blur_limit=3, p=0.2),
    dict(type="ISONoise", p=0.2),
    dict(type="ImageCompression", quality_lower=15, quality_upper=30, p=0.25),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=384, edge='short', backend='pillow'),
    dict(type='Albu', transforms=albu_train_transforms),
   #dict(type="Cutout", shape=(8, 8), pad_val=8, prob=0.05),
    dict(
        type='RandomResizedCrop',
        scale=384,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=384,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=384),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/rubble/train',
        classes=classes,
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/rubble/val',
        classes=classes,
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/rubble/test',
        classes=classes,
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# Specify the evaluation metric for validation and testing.
val_evaluator = dict(type='Accuracy', topk=(1, 2))
test_evaluator = val_evaluator

# ---- RUNTIME SETTINGS ----
# Output training log every 10 iterations.
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, save_best="auto", max_keep_ckpts=2),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)

# If you want to ensure reproducibility, set a random seed. And enable the
# deterministic option in cuDNN to further ensure reproducibility, but it may
# reduce the training speed.
randomness = dict(seed=None, deterministic=False)
