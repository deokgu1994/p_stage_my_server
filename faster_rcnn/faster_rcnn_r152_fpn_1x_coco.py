_base_ = 'faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        depth=152,
        init_cfg=dict(type='Pretrained',
                      checkpoint='https://download.pytorch.org/models/resnet152-b121ed2d.pth')))
