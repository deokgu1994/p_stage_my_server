_base_ = [
    'faster_rcnn_r50_fpn_1x_coco.py'
]

model = dict(
    backbone=dict(
        depth=152,
        init_cfg=dict(type='Pretrained',
                      checkpoint='https://download.pytorch.org/models/resnet152-b121ed2d.pth'),
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
