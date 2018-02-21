from lib.models import ssd
from lib.models import rfb
from lib.models import fssd
from lib.models import rfb_lite
models_map = {
                'ssd': ssd.build_ssd,
                'rfb': rfb.build_rfb,
                'rfb_lite': rfb_lite.build_rfb_lite,
                'fssd': fssd.build_fssd,
            }

def gen_model_fn(name):
    """Returns a build ssd func.

    Args:
    name: The type of the ssd.

    Returns:
    base: base model_fn

    Raises:
    ValueError: If network `name` is not recognized.
    """
    if name not in models_map:
        raise ValueError('Type of ssd unknown %s' % name)
    func = models_map[name]
    return func