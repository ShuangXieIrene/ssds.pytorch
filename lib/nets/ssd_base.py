import torch
import torch.nn as nn

def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [24, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)

# def multibox(cfg, num_classes):
#     vgg = vgg(base[str(size)], 3)
#     #net = net_factory(cfg[net])
#     extra_layers = add_extras(cfg, vgg[-1].out_channels)
#     loc_layers = []
#     conf_layers = []
#     vgg_source = [24, -2]
#     for k, v in enumerate(vgg_source):
#         loc_layers += [nn.Conv2d(vgg[v].out_channels,
#                                  cfg[k] * 4, kernel_size=3, padding=1)]
#         conf_layers += [nn.Conv2d(vgg[v].out_channels,
#                         cfg[k] * num_classes, kernel_size=3, padding=1)]
#     for k, v in enumerate(extra_layers[1::2], 2):
#         loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
#                                  * 4, kernel_size=3, padding=1)]
#         conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
#                                   * num_classes, kernel_size=3, padding=1)]
#     return vgg, extra_layers, (loc_layers, conf_layers)

extras = {
    'vgg16': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
}
mbox = {
    'vgg16': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
}