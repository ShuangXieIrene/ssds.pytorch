import torch
import cv2
import numpy as np
import math
from itertools import product as product

def images_to_writer(writer, images, prefix='image', names='image', epoch=0):
    if isinstance(names, str):
        names = [names+'_{}'.format(i) for i in range(len(images))]

    for image, name in zip(images, names):
        writer.add_image('{}/{}'.format(prefix, name), image, epoch)


def to_grayscale(image):
    """
    input is (d,w,h)
    converts 3D image tensor to grayscale images corresponding to each channel
    """
    # print(image.shape)
    channel = image.shape[0]
    image = torch.sum(image, dim=0)
    # print(image.shape)
    image = torch.div(image, channel)
    # print(image.shape)
    # assert False
    return image


def to_image_size(feature, target_img):
    height, width, _ = target_img.shape
    resized_feature = cv2.resize(feature, (width, height)) 
    return resized_feature


def features_to_grid(features):
    num, height, width, channel = (len(features), len(features[0]), len(features[0][0]), len(features[0][0]))
    rows = math.ceil(np.sqrt(num))
    output = np.zeros([rows*(height+2),rows*(width+2), 3],dtype=np.float32)

    for i, feature in enumerate(features):
        row = i % rows
        col = math.floor(i / rows)
        output[row*(2+height)+1:(row+1)*(2+height)-1, col*(2+width)+1:(col+1)*(2+width)-1] = feature

    return output
    

def viz_feature_maps(writer, feature_maps, module_name='base', epoch=0, prefix='module_feature_maps'):
    feature_map_visualization = []
    for i in feature_maps:
        i = i.squeeze(0)
        temp = to_grayscale(i)
        feature_map_visualization.append(temp.data.cpu().numpy())
        
    names, feature_map_heatmap = [], []
    for i, feature_map in enumerate(feature_map_visualization):
        feature_map = (feature_map * 255)
        heatmap = cv2.applyColorMap(feature_map.astype(np.uint8), cv2.COLORMAP_JET)
        feature_map_heatmap.append(heatmap[..., ::-1])
        names.append('{}.{}'.format(module_name, i))

    images_to_writer(writer, feature_map_heatmap, prefix, names, epoch)

def viz_grads(writer, model, feature_maps, target_image, target_mean, module_name='base', epoch=0, prefix='module_grads'):
    grads_visualization = []
    names = []
    for i, feature_map in enumerate(feature_maps):
        model.zero_grad()
        
        # print()
        feature_map.backward(torch.Tensor(np.ones(feature_map.size())), retain_graph=True)
        # print(target_image.grad)
        grads = target_image.grad.data.clamp(min=0).squeeze(0).permute(1,2,0)
        # print(grads)
        # assert False
        grads_visualization.append(grads.cpu().numpy()+target_mean)
        names.append('{}.{}'.format(module_name, i))
    
    images_to_writer(writer, grads_visualization, prefix, names, epoch)

def viz_module_feature_maps(writer, module, input_image, module_name='base', epoch=0, mode='one', prefix='module_feature_maps'):
    output_image = input_image
    feature_maps = []

    for i, layer in enumerate(module):
        output_image = layer(output_image)
        feature_maps.append(output_image)

    if mode is 'grid':
        pass
    elif mode is 'one':
        viz_feature_maps(writer, feature_maps, module_name, epoch, prefix)

    return output_image


def viz_module_grads(writer, model, module, input_image, target_image, target_mean, module_name='base', epoch=0, mode='one', prefix='module_grads'):
    output_image = input_image
    feature_maps = []

    for i, layer in enumerate(module):
        output_image = layer(output_image)
        feature_maps.append(output_image)

    if mode is 'grid':
        pass
    elif mode is 'one':
        viz_grads(writer, model, feature_maps, target_image, target_mean, module_name, epoch, prefix)
    
    return output_image

def viz_prior_box(writer, prior_box, image=None, epoch=0):  
    if isinstance(image, type(None)):
        image = np.random.random((prior_box.image_size[0], prior_box.image_size[1], 3))
    elif isinstance(image, str):
        image = cv2.imread(image, -1)
    image = cv2.resize(image, (prior_box.image_size[0], prior_box.image_size[1]))
    
    for k, f in enumerate(prior_box.feature_maps):
        bbxs = []
        image_show = image.copy()
        for i, j in product(range(f[0]), range(f[1])):
            cx = j * prior_box.steps[k][1] + prior_box.offset[k][1]
            cy = i * prior_box.steps[k][0] + prior_box.offset[k][0]

            # aspect_ratio: 1 Min size
            s_k = prior_box.scales[k]
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

        scale = [prior_box.image_size[1], prior_box.image_size[0], prior_box.image_size[1], prior_box.image_size[0]]
        bbxs = np.array(bbxs).reshape((-1, 4))
        archors = bbxs[:, :2] * scale[:2]
        bbxs = np.hstack((bbxs[:, :2] - bbxs[:, 2:4]/2, bbxs[:, :2] + bbxs[:, 2:4]/2)) * scale
        archors = archors.astype(np.int32)
        bbxs = bbxs.astype(np.int32)

        for archor, bbx in zip(archors, bbxs):
            cv2.circle(image_show,(archor[0],archor[1]), 2, (0,0,255), -1)
            if archor[0] == archor[1]:
                cv2.rectangle(image_show, (bbx[0], bbx[1]), (bbx[2], bbx[3]), (0, 255, 0), 1)

        writer.add_image('example_prior_boxs/feature_map_{}'.format(k), image_show, epoch)


def add_pr_curve_raw(writer, tag, precision, recall, epoch=0):
    num_thresholds = len(precision)
    writer.add_pr_curve_raw(
        tag=tag,
        true_positive_counts = -np.ones(num_thresholds),
        false_positive_counts = -np.ones(num_thresholds),
        true_negative_counts = -np.ones(num_thresholds),
        false_negative_counts = -np.ones(num_thresholds),
        precision = precision,
        recall = recall,
        global_step = epoch,
        num_thresholds = num_thresholds
    )


def viz_pr_curve(writer, precision, recall, epoch=0):
    for i, (_prec, _rec) in enumerate(zip(precision, recall)):
        # _prec, _rec = prec, rec
        num_thresholds = min(500, len(_prec))
        if num_thresholds != len(_prec):
            gap = int(len(_prec) / num_thresholds)
            _prec = np.append(_prec[::gap], _prec[-1])
            _rec  = np.append(_rec[::gap], _rec[-1])
            num_thresholds = len(_prec)
        # the pr_curve_raw_data_pb() needs the a ascending precisions array and a descending recalls array
        _prec.sort()
        _rec[::-1].sort()
        # TODO: need to change i to the name of the class
        # 0 is the background class as default
        add_pr_curve_raw(
            writer=writer, tag='pr_curve/class_{}'.format(i+1), precision = _prec, recall = _rec, epoch = epoch )


def viz_archor_strategy(writer, sizes, labels, epoch=0):
    ''' generate archor strategy for all classes
    '''

    # merge all data into one 
    height, width, max_size, min_size, aspect_ratio, label = [list() for _ in range(6)]
    for _size, _label in zip(sizes[1:], labels[1:]):
        _height, _width, _max_size, _min_size, _aspect_ratio = [list() for _ in range(5)]
        for size in _size:
            _height += [size[0]]
            _width  += [size[1]]
            _max_size += [max(size)]
            _min_size += [min(size)]
            _aspect_ratio += [size[0]/size[1] if size[0] < size[1] else size[1]/size[0]]
        height += _height
        width += _width
        max_size += _max_size
        min_size += _min_size
        aspect_ratio += _aspect_ratio
        label += _label
    
    height, width, max_size, min_size, aspect_ratio = \
        np.array(height), np.array(width), np.array(max_size), np.array(min_size), np.array(aspect_ratio)   
    matched_height, matched_width, matched_max_size, matched_min_size, matched_aspect_ratio = \
        height[label], width[label], max_size[label], min_size[label], aspect_ratio[label]

    num_thresholds = 100
    # height, width, max_size, min_size, aspect_ratio = \
    height.sort(), width.sort(), max_size.sort(), min_size.sort(), aspect_ratio.sort()
    # matched_height, matched_width, matched_max_size, matched_min_size, matched_aspect_ratio = \
    matched_height.sort(), matched_width.sort(), matched_max_size.sort(), matched_min_size.sort(), matched_aspect_ratio.sort()

    x_axis = np.arange(num_thresholds)[::-1]/num_thresholds + 0.5 / num_thresholds

    # height 
    gt_y, _ = np.histogram(height, bins=num_thresholds, range=(0.0, 1.0))
    gt_y = np.clip( gt_y[::-1]/len(height), 1e-8, 1.0)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/height_distribute_gt', precision = gt_y, recall = x_axis, epoch = epoch )
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/height_distribute_gt_normalized', precision = gt_y/max(gt_y), recall = x_axis, epoch = epoch )
    
    matched_y, _ = np.histogram(matched_height, bins=num_thresholds,range=(0.0, 1.0))
    matched_y = np.clip( matched_y[::-1]/len(height), 1e-8, 1.0)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/height_distribute_matched', precision = matched_y/gt_y, recall = x_axis, epoch = epoch )

    # width
    gt_y, _ = np.histogram(width, bins=num_thresholds, range=(0.0, 1.0))
    gt_y = np.clip( gt_y[::-1]/len(width), 1e-8, 1.0)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/width_distribute_gt', precision = gt_y, recall = x_axis, epoch = epoch )
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/width_distribute_gt_normalized', precision = gt_y/max(gt_y), recall = x_axis, epoch = epoch )
    
    matched_y, _ = np.histogram(matched_width, bins=num_thresholds,range=(0.0, 1.0))
    matched_y = np.clip( matched_y[::-1]/len(width), 1e-8, 1.0)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/width_distribute_matched', precision = matched_y/gt_y, recall = x_axis, epoch = epoch )

    # max_size
    gt_y, _ = np.histogram(max_size, bins=num_thresholds, range=(0.0, 1.0))
    gt_y = np.clip( gt_y[::-1]/len(max_size), 1e-8, 1.0)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/max_size_distribute_gt', precision = gt_y, recall = x_axis, epoch = epoch )
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/max_size_distribute_gt_normalized', precision = gt_y/max(gt_y), recall = x_axis, epoch = epoch )
    
    matched_y, _ = np.histogram(matched_max_size, bins=num_thresholds,range=(0.0, 1.0))
    matched_y = np.clip( matched_y[::-1]/len(max_size), 1e-8, 1.0)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/max_size_distribute_matched', precision = matched_y/gt_y, recall = x_axis, epoch = epoch )

    # min_size
    gt_y, _ = np.histogram(min_size, bins=num_thresholds, range=(0.0, 1.0))
    gt_y = np.clip( gt_y[::-1]/len(min_size), 1e-8, 1.0)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/min_size_distribute_gt', precision = gt_y, recall = x_axis, epoch = epoch )
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/min_size_distribute_gt_normalized', precision = gt_y/max(gt_y), recall = x_axis, epoch = epoch )
    
    matched_y, _ = np.histogram(matched_min_size, bins=num_thresholds,range=(0.0, 1.0))
    matched_y = np.clip( matched_y[::-1]/len(min_size), 1e-8, 1.0)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/min_size_distribute_matched', precision = matched_y/gt_y, recall = x_axis, epoch = epoch )
    
    # aspect_ratio
    gt_y, _ = np.histogram(aspect_ratio, bins=num_thresholds, range=(0.0, 1.0))
    gt_y = np.clip( gt_y[::-1]/len(aspect_ratio), 1e-8, 1.0)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/aspect_ratio_distribute_gt', precision = gt_y, recall = x_axis, epoch = epoch )
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/aspect_ratio_distribute_gt_normalized', precision = gt_y/max(gt_y), recall = x_axis, epoch = epoch )
    
    matched_y, _ = np.histogram(matched_aspect_ratio, bins=num_thresholds,range=(0.0, 1.0))
    matched_y = np.clip( matched_y[::-1]/len(aspect_ratio), 1e-8, 1.0)
    add_pr_curve_raw(
        writer=writer, tag='archor_strategy/aspect_ratio_distribute_matched', precision = matched_y/gt_y, recall = x_axis, epoch = epoch )