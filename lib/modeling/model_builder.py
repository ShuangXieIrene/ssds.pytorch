
# ssds part
from lib.modeling.ssds import ssd
from lib.modeling.ssds import ssd_lite
from lib.modeling.ssds import rfb
from lib.modeling.ssds import rfb_lite
from lib.modeling.ssds import fssd

ssds_map = {
                'ssd': ssd.build_ssd,
                'ssd_lite': ssd_lite.build_ssd_lite,
                'rfb': rfb.build_rfb,
                'rfb_lite': rfb_lite.build_rfb_lite,
                'fssd': fssd.build_fssd,
            }

# nets part
from lib.modeling.nets import vgg
from lib.modeling.nets import resnet
from lib.modeling.nets import mobilenet

networks_map = {
                    'vgg16': vgg.vgg16,
                    'resnet_101': resnet.resnet_101,
                    'mobilenet_v1': mobilenet.mobilenet_v1,
                    'mobilenet_v1_075': mobilenet.mobilenet_v1_075,
                    'mobilenet_v1_050': mobilenet.mobilenet_v1_050,
                    'mobilenet_v1_025': mobilenet.mobilenet_v1_025,
                    'mobilenet_v2': mobilenet.mobilenet_v2,
                    'mobilenet_v2_075': mobilenet.mobilenet_v2_075,
                    'mobilenet_v2_050': mobilenet.mobilenet_v2_050,
                    'mobilenet_v2_025': mobilenet.mobilenet_v2_025,
               }

from lib.layers.functions.prior_box import PriorBox
from torch.autograd import Variable

def create_model(cfg):
    '''
    '''
    #
    base = networks_map[cfg.NETS]
    number_box=[2+2*len(aspect_ratios) for aspect_ratios in cfg.ASPECT_RATIOS]
    model = ssds_map[cfg.SSDS](base=base, feature_layer=cfg.FEATURE_LAYER, mbox=number_box, num_classes=cfg.NUM_CLASSES)
    #
    feature_maps = model._forward_features_size(cfg.IMAGE_SIZE)
    print('==>Feature map size:')
    print(feature_maps)
    priorbox = PriorBox(image_size=cfg.IMAGE_SIZE, feature_maps=feature_maps, aspect_ratios=cfg.ASPECT_RATIOS, 
                    scale=cfg.SIZES, archor_stride=cfg.STEPS, clip=cfg.CLIP)
    # priors = Variable(priorbox.forward(), volatile=True)

    return model, priorbox