_base_ = [
    'faster_rcnn_r152_fpn_dconv_c3-c5_1x_coco.py'
]

model = dict(
    backbone=dict(
        depth=152,
        init_cfg=dict(type='Pretrained',
                      checkpoint='https://download.pytorch.org/models/resnet152-b121ed2d.pth'),
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
