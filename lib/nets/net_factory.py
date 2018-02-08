from lib.nets import vgg
from lib.nets import mobilenet

networks_map = {
                    'vgg16': vgg.vgg16,
                    'mobilenet_v1': mobilenet.mobilenet_v1,
                    'mobilenet_v1_075': mobilenet.mobilenet_v1_075,
                    'mobilenet_v1_050': mobilenet.mobilenet_v1_050,
                    'mobilenet_v1_025': mobilenet.mobilenet_v1_025,
                    'mobilenet_v2': mobilenet.mobilenet_v2,
                    'mobilenet_v2_075': mobilenet.mobilenet_v2_075,
                    'mobilenet_v2_050': mobilenet.mobilenet_v2_050,
                    'mobilenet_v2_025': mobilenet.mobilenet_v2_025,
               }

def gen_base_fn(name):
    """Returns a base network for ssd.

    Args:
    name: The name of the network.

    Returns:
    base: base network_fn

    Raises:
    ValueError: If network `name` is not recognized.
    """
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    func = networks_map[name]
    return func