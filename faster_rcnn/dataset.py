# dataset settings
dataset_type = "CocoDataset"
data_root = "/opt/ml/detection/dataset/"
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# RGB Mean: [123.6506697  117.39730243 110.07542563]
# RGB Standard Deviation: [54.03457934 53.36968771 54.78390763]
# RGB Mean: [0.48490459 0.46038158 0.43166834]  /255
# RGB Standard Deviation: [0.21190031 0.20929289 0.21483885]
img_norm_cfg = dict(
    mean=[123.650, 117.39, 110.07], std=[54.035, 53.36, 54.783], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            # [
            #     # dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
            #     dict(
            #         type='MixUp',
            #         img_scale=(1024, 1024),
            #         ratio_range=(0.8, 1.6),
            #         pad_val=114.0)
            #     # dict(
            #     #     type='RandomAffine',
            #     #     scaling_ratio_range=(0.1, 2),
            #     #     border=(-img_scale[0] // 2, -img_scale[1] // 2)),
            # ],
            [
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(800, 1024),
                    allow_negative_crop=True),
                dict(
                    type='Resize',
                    img_scale=[(480, 1024), (512, 1024), (544, 1024),
                               (576, 1024), (608, 1024), (640, 1024),
                               (672, 1024), (704, 1024), (736, 1024),
                               (768, 1024), (800, 1024)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ],
        ]),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=6,  # batch_size
    workers_per_gpu=4,  # dataload만들데 num_works
    train=dict(
        type=dataset_type,
        ann_file=data_root + "train.json",
        img_prefix=data_root,
        classes = classes,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "train.json",
        img_prefix=data_root,
        classes = classes,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "test.json",
        img_prefix=data_root,
        classes = classes,
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=3, metric="bbox")
