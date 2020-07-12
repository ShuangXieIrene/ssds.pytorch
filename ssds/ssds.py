import torch

from .core import checkpoint, config
from .modeling import model_builder

class SSDDetector(object):
    def __init__(self, cfg_file, is_print=False):
        # Config
        cfg = config.cfg_from_file(cfg_file)

        # Build model
        print('===> Building model')
        self.model = model_builder.create_model(cfg.MODEL)
        if is_print: print('Model architectures:\n{}\n'.format(self.model))
        self.anchors = model_builder.create_anchors(cfg.MODEL, self.model, cfg.MODEL.IMAGE_SIZE, is_print)
        self.decoder = model_builder.create_decoder(cfg.POST_PROCESS)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if cfg.RESUME_CHECKPOINT:
            print('Loading initial model weights from {:s}'.format(cfg.RESUME_CHECKPOINT))
            checkpoint.resume_checkpoint(self.model, cfg.RESUME_CHECKPOINT, '')
        self.model.eval().to(self.device)

        self.image_size = tuple(cfg.MODEL.IMAGE_SIZE)
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.mean = cfg.DATASET.PREPROC.MEAN
        self.std  = cfg.DATASET.PREPROC.STD

    def __call__(self, imgs):
        pick1st=False
        if len(imgs.shape) == 3:
            imgs = imgs[None, ...]
            pick1st = True
        if len(imgs.shape) != 4:
            raise AssertionError("image dims has to be 3 or 4")
        if imgs.shape[3] == 3:
            imgs = imgs.transpose(0,3,1,2)

        imgs_tensor = torch.Tensor(imgs).to(self.device)
        imgs_tensor = (imgs_tensor - self.mean)/self.std

        loc, conf = self.model(imgs_tensor)
        detections = self.decoder(loc, conf, self.anchors)
        out_scores, out_boxes, out_classes = (d.cpu().detach().numpy() for d in detections)
    
        if pick1st:
            return out_scores[0], out_boxes[0].astype(int), out_classes[0].astype(int)
        else:
            return out_scores, out_boxes.astype(int), out_classes.astype(int)
