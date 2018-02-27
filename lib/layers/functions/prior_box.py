from __future__ import division
import torch
from math import sqrt as sqrt
from itertools import product as product
# class PriorBox(object):
#     """Compute priorbox coordinates in center-offset form for each source
#     feature map.
#     Note:
#     This 'layer' has changed between versions of the original SSD
#     paper, so we include both versions, but note v2 is the most tested and most
#     recent version of the paper.

#     """
#     def __init__(self, cfg):
#         super(PriorBox, self).__init__()
#         self.image_size = cfg['MIN_DIM']
#         # number of priors for feature map location (either 4 or 6)
#         self.num_priors = len(cfg['ASPECT_RATIOS'])
#         self.variance = cfg['VARIANCE'] or [0.1]
#         self.feature_maps = cfg['FEATURE_MAPS']
#         self.min_sizes = cfg['MIN_SIZES']
#         self.max_sizes = cfg['MAX_SIZES']
#         self.steps = cfg['STEPS']
#         self.aspect_ratios = cfg['ASPECT_RATIOS']
#         self.clip = cfg['CLIP']
#         for v in self.variance:
#             if v <= 0:
#                 raise ValueError('Variances must be greater than 0')

#     def forward(self):
#         mean = []
#         for k, f in enumerate(self.feature_maps):
#             for i, j in product(range(f), repeat=2):
#                 f_k = self.image_size / self.steps[k]
#                 cx = (j + 0.5) / f_k
#                 cy = (i + 0.5) / f_k

#                 s_k = self.min_sizes[k]/self.image_size
#                 mean += [cx, cy, s_k, s_k]

#                 # aspect_ratio: 1
#                 # rel size: sqrt(s_k * s_(k+1))
#                 s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
#                 mean += [cx, cy, s_k_prime, s_k_prime]

#                 # rest of aspect ratios
#                 for ar in self.aspect_ratios[k]:
#                     mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
#                     mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
#             print(f)
#             print(len(mean))
#             assert False
#         # back to torch land
#         output = torch.Tensor(mean).view(-1, 4)
#         if self.clip:
#             output.clamp_(max=1, min=0)
#         return output


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.

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
            for i, j in product(range(f[0]), range(f[1])): #TODO: check the order of i,j
                cx = j * self.steps[k][1] + self.offset[k][1]
                cy = i * self.steps[k][0] + self.offset[k][0]

                # aspect_ratio: 1 Min size
                s_k = self.scales[k]
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1 Max size
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * self.scales[k+1])
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    ar_sqrt = sqrt(ar)
                    mean += [cx, cy, s_k*ar_sqrt, s_k/ar_sqrt]
                    mean += [cx, cy, s_k/ar_sqrt, s_k*ar_sqrt]
            
            # if k == 4:
            #     print(f)
            #     print(mean[l:])
            #     print(len(mean[l:])/4)
            #     assert False
            # l = len(mean)
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

    def draw_box(self, writer, image=None):
        import numpy as np
        import cv2

        if isinstance(image, type(None)):
            image = np.random.random((self.image_size[0], self.image_size[1], 3))
        elif isinstance(image, str):
            image = cv2.imread(image, -1)
        image = cv2.resize(image, (self.image_size[0], self.image_size[1]))
        
        for k, f in enumerate(self.feature_maps):
            bbxs = []
            image_show = image.copy()
            for i, j in product(range(f[0]), range(f[1])): #TODO: check the order of i,j
                cx = j * self.steps[k][1] + self.offset[k][1]
                cy = i * self.steps[k][0] + self.offset[k][0]

                # aspect_ratio: 1 Min size
                s_k = self.scales[k]
                bbxs += [cx, cy, s_k, s_k]

                # # aspect_ratio: 1 Max size
                # # rel size: sqrt(s_k * s_(k+1))
                # s_k_prime = sqrt(s_k * self.scales[k+1])
                # bbxs += [cx, cy, s_k_prime, s_k_prime]

                # # rest of aspect ratios
                # for ar in self.aspect_ratios[k]:
                #     ar_sqrt = sqrt(ar)
                #     bbxs += [cx, cy, s_k*ar_sqrt, s_k/ar_sqrt]
                #     bbxs += [cx, cy, s_k/ar_sqrt, s_k*ar_sqrt]

            scale = [self.image_size[1], self.image_size[0], self.image_size[1], self.image_size[0]]
            bbxs = np.array(bbxs).reshape((-1, 4))
            archors = bbxs[:, :2] * scale[:2]
            bbxs = np.hstack((bbxs[:, :2] - bbxs[:, 2:4]/2, bbxs[:, :2] + bbxs[:, 2:4]/2)) * scale
            archors = archors.astype(np.int32)
            bbxs = bbxs.astype(np.int32)

            for archor, bbx in zip(archors, bbxs):
                cv2.circle(image_show,(archor[0],archor[1]), 2, (0,0,255), -1)
                if archor[0] == archor[1]:
                    cv2.rectangle(image_show, (bbx[0], bbx[1]), (bbx[2], bbx[3]), (0, 255, 0), 1)

            writer.add_image('example_prior_boxs/feature_map_{}'.format(k), image_show, 0)

