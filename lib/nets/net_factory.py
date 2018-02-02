from lib.nets import vgg

networks_map = {
                    'vgg16': vgg.create_vgg16_base,
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