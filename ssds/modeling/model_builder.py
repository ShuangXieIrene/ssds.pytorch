import torch
from collections import OrderedDict

from . import ssds, nets
from .layers.box import generate_anchors, configure_ratio_scale
from .layers.decoder import Decoder


def create_model(cfg):
    """ create the model based on the config files
    Returns:
        torch ssds model with backbone as net
    """
    ratios, scales = configure_ratio_scale(len(cfg.SIZES), cfg.ASPECT_RATIOS, cfg.SIZES)
    number_box = [len(r) * len(s) for r, s in zip(ratios, scales)]
    nets_outputs, extras, head = getattr(ssds, cfg.SSDS).add_extras(
        feature_layer=cfg.FEATURE_LAYER, mbox=number_box, num_classes=cfg.NUM_CLASSES
    )
    model = getattr(ssds, cfg.SSDS)(
        backbone=getattr(nets, cfg.NETS)(
            outputs=nets_outputs, num_images=cfg.NUM_IMAGES
        ),
        extras=extras,
        head=head,
        num_classes=cfg.NUM_CLASSES,
    )
    return model


def create_anchors(cfg, model, image_size, visualize=False):
    """ current version for generate the anchor, only generate the default anchor for each feature map layers
    Returns:
        anchors: OrderedDict(key=stride, value=default_anchors)
    """
    model.eval()
    with torch.no_grad():
        x = torch.rand(
            (1, 3, image_size[0], image_size[1]), device=next(model.parameters()).device
        )
        conf = model(x)[-1]
        strides = [x.shape[-1] // c.shape[-1] for c in conf]

    ratios, scales = configure_ratio_scale(len(strides), cfg.ASPECT_RATIOS, cfg.SIZES)
    anchors = OrderedDict(
        [
            (strides[i], generate_anchors(strides[i], ratios[i], scales[i]))
            for i in range(len(strides))
        ]
    )
    if visualize:
        print("Anchor Boxs (width, height)")
        [
            print("Stride {}: {}".format(k, (v[:, 2:] - v[:, :2] + 1).int().tolist()))
            for k, v in anchors.items()
        ]
    return anchors


def create_decoder(cfg):
    r""" Generate decoder based on the cfg.POST_PROCESS. 
    
    The generated decoder is the object of class Decoder, check more details by :class:`ssds.modeling.layers.decoder.Decoder`.
    
    Args:
        cfg: defined cfg.POST_PROCESS
    """
    return Decoder(
        cfg.SCORE_THRESHOLD,
        cfg.IOU_THRESHOLD,
        cfg.MAX_DETECTIONS,
        cfg.MAX_DETECTIONS_PER_LEVEL,
        cfg.RESCORE_CENTER,
        cfg.USE_DIOU,
    )
