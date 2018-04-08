from __future__ import division
import torch
from math import sqrt as sqrt
from itertools import product as product

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, image_size, feature_maps, aspect_ratios, scale, archor_stride=None, archor_offest=None, clip=True):
        super(PriorBox, self).__init__()
        self.image_size = image_size #[height, width]
        self.feature_maps = feature_maps #[(height, width), ...]
        self.aspect_ratios = aspect_ratios
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(aspect_ratios)
        self.clip = clip
        # scale value
        if isinstance(scale[0], list):
            # get min of the result
            self.scales = [min(s[0] / self.image_size[0], s[1] / self.image_size[1]) for s in scale]
        elif isinstance(scale[0], float) and len(scale) == 2:
            num_layers = len(feature_maps)
            min_scale, max_scale = scale
            self.scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1) for i in range(num_layers)] + [1.0]
        
        if archor_stride:
            self.steps = [(steps[0] / self.image_size[0], steps[1] / self.image_size[1]) for steps in archor_stride] 
        else:
            self.steps = [(1/f_h, 1/f_w) for f_h, f_w in feature_maps]

        if archor_offest:
            self.offset = [[offset[0] / self.image_size[0], offset[1] * self.image_size[1]] for offset in archor_offest] 
        else:
            self.offset = [[steps[0] * 0.5, steps[1] * 0.5] for steps in self.steps] 

    def forward(self):
        mean = []
        # l = 0
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f[0]), range(f[1])):
                cx = j * self.steps[k][1] + self.offset[k][1]
                cy = i * self.steps[k][0] + self.offset[k][0]
                s_k = self.scales[k]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    if isinstance(ar, int):
                        if ar == 1:
                            # aspect_ratio: 1 Min size
                            mean += [cx, cy, s_k, s_k]

                            # aspect_ratio: 1 Max size
                            # rel size: sqrt(s_k * s_(k+1))
                            s_k_prime = sqrt(s_k * self.scales[k+1])
                            mean += [cx, cy, s_k_prime, s_k_prime]
                        else:
                            ar_sqrt = sqrt(ar)
                            mean += [cx, cy, s_k*ar_sqrt, s_k/ar_sqrt]
                            mean += [cx, cy, s_k/ar_sqrt, s_k*ar_sqrt]
                    elif isinstance(ar, list):
                        mean += [cx, cy, s_k*ar[0], s_k*ar[1]]
        #     print(f, self.aspect_ratios[k])
        # assert False
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output