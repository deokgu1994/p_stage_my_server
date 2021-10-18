import torch.nn as nn
import torch.optim as optim
from torchvision import models

# 최고 성능 cspdarkdet53m
def custom_efficientdet(det_name, num_classes, image_size, checkpoint_path=None):
    # Effdet config
    # https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/config/model_config.py
    """
        Returns a default detection configs.
        h = OmegaConf.create()

        # model name.
        h.name = 'tf_efficientdet_d1'

        h.backbone_name = 'tf_efficientnet_b1'
        h.backbone_args = None  # FIXME sort out kwargs vs config for backbone creation
        h.backbone_indices = None

        # model specific, input preprocessing parameters
        h.image_size = (640, 640)

        # dataset specific head parameters
        h.num_classes = 90

        # feature + anchor config
        h.min_level = 3
        h.max_level = 7
        h.num_levels = h.max_level - h.min_level + 1
        h.num_scales = 3
        h.aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        # ratio w/h: 2.0 means w=1.4, h=0.7. Can be computed with k-mean per dataset.
        # aspect ratios can be specified as below too, pairs will be calc as sqrt(val), 1/sqrt(val)
        #h.aspect_ratios = [1.0, 2.0, 0.5]
        h.anchor_scale = 4.0

        # FPN and head config
        h.pad_type = 'same'  # original TF models require an equivalent of Tensorflow 'SAME' padding
        h.act_type = 'swish'
        h.norm_layer = None  # defaults to batch norm when None
        h.norm_kwargs = dict(eps=.001, momentum=.01)
        h.box_class_repeats = 3
        h.fpn_cell_repeats = 3
        h.fpn_channels = 88
        h.separable_conv = True
        h.apply_resample_bn = True
        h.conv_after_downsample = False
        h.conv_bn_relu_pattern = False
        h.use_native_resize_op = False
        h.downsample_type = 'max'
        h.upsample_type = 'nearest'
        h.redundant_bias = True  # original TF models have back to back bias + BN layers, not necessary!
        h.head_bn_level_first = False  # change order of BN in head repeat list of lists, True for torchscript compat
        h.head_act_type = None  # activation for heads, same as act_type if None

        h.fpn_name = None
        h.fpn_config = None
        h.fpn_drop_path_rate = 0.  # No stochastic depth in default. NOTE not currently used, unstable training

        # classification loss (used by train bench)
        h.alpha = 0.25
        h.gamma = 1.5
        h.label_smoothing = 0.  # only supported if legacy_focal == False, haven't produced great results
        h.legacy_focal = False  # use legacy focal loss (less stable, lower memory use in some cases)
        h.jit_loss = False  # torchscript jit for loss fn speed improvement, can impact stability and/or increase mem usage

        # localization loss (used by train bench)
        h.delta = 0.1
        h.box_loss_weight = 50.0

        # nms
        h.soft_nms = False  # use soft-nms, this is incredibly slow
        h.max_detection_points = 5000  # max detections for post process, input to NMS
        h.max_det_per_image = 100  # max detections per image limit, output of NMS
    """

    from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
    from effdet.efficientdet import HeadNet

    config = get_efficientdet_config(det_name)
    config.num_classes = num_classes
    config.image_size = (512,512)
    
    config.soft_nms = True
    config.max_det_per_image = 40
    
    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        
    return DetBenchTrain(net)


    