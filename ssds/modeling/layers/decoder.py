import torch
from .box import decode, nms


class Decoder(object):
    r"""
    class Decoder contains the decoder func and nms func
    
    * decoder
        decoder is used to decode the boxes from loc and conf feature map, check :meth:`ssds.modeling.layers.box.decode` for more details.
    * nms
        nms is used to filter the decoded boxes by its confidence and box location, check :meth:`ssds.modeling.layers.box.nms` for more details.
    """

    def __init__(
        self, conf_threshold, nms_threshold, top_n, top_n_per_level, rescore, use_diou
    ):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_n = top_n
        self.top_n_per_level = top_n_per_level
        self.rescore = rescore
        self.use_diou = use_diou

    def __call__(self, loc, conf, anchors):
        r"""
        Decode and filter boxes

        Returns:
            out_scores,  (batch, top_n)

            out_boxes,   (batch, top_n, 4) with ltrb format

            out_classes, (batch, top_n)
        """
        decoded = [
            decode(
                c,
                l,
                stride,
                self.conf_threshold,
                self.top_n_per_level,
                anchor,
                rescore=self.rescore,
            )
            for l, c, (stride, anchor) in zip(loc, conf, anchors.items())
        ]
        decoded = [torch.cat(tensors, 1) for tensors in zip(*decoded)]
        return nms(*decoded, self.nms_threshold, self.top_n, using_diou=self.use_diou)
