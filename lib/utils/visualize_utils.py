import torch
import cv2
import numpy as np
import math
from itertools import product as product

def to_grayscale(image):
    """
    input is (d,w,h)
    converts 3D image tensor to grayscale images corresponding to each channel
    """
    image = torch.sum(image, dim=0)
    image = torch.div(image, image.shape[0])
    return image

def to_image_size(feature, target_img):
    height, width, _ = target_img.shape
    resized_feature = cv2.resize(feature, (width, height)) 
    return resized_feature

def to_heatmap(gray_img):
    heatmap_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
    return heatmap_img

def features_to_grid(features):
    num, height, width, channel = (len(features), len(features[0]), len(features[0][0]), len(features[0][0]))
    rows = math.ceil(np.sqrt(num))
    output = np.zeros([rows*(height+2),rows*(width+2), 3],dtype=np.float32)

    for i, feature in enumerate(features):
        row = i % rows
        col = math.floor(i / rows)
        output[row*(2+height)+1:(row+1)*(2+height)-1, col*(2+width)+1:(col+1)*(2+width)-1] = feature

    return output
    

def base_layer_outputs(model, image):
    modulelist = list(model.base.modules())
    outputs, output_im, names = [list() for _ in range(3)]
    for i, layer in enumerate(modulelist): 
        image = layer(image)
        outputs.append(image)
        names.append('base.{}.{}'.format(i, str(layer)))

    for i in outputs:
        i = i.squeeze(0)
        temp = to_grayscale(i)
        output_im.append(temp.data.cpu().numpy())
    return output_im, names


def all_feature_maps_outputs(feature_map):
    features = []
    names = str(feature_map)
    for i in range(feature_map.shape[0]):
        features.append(feature_map[i,:,:])
    return features, names

def one_feature_maps_outputs(feature_map):
    names = str(feature_map)
    temp = feature_map.squeeze(0)
    temp = to_grayscale(temp)
    feature = temp.data.cpu().numpy()
    return feature, names

def images_to_writer(writer, images, prefix='image', names='image', epoch=0):
    if isinstance(names, str):
        names = [names+'_{}'.format(i) for i in range(len(images))]

    for image, name in zip(images, names):
        writer.add_image('{}/{}'.format(prefix, name), image, epoch)
    
def viz_base_layer(writer, model, image, epoch=0):
    feature_maps, names = base_layer_outputs(model, image)
    for feature_map in feature_maps:
        feature_map_heatmap = to_heatmap(feature_map)
    images_to_writer(writer, feature_map_heatmap, 'base_layers', names, epoch=0)

def viz_feature_extractor(writer, feature_extractor, epoch=0):
    features, names = [list() for _ in range(2)]
    for feature_maps in feature_extractor:
        feature, name = one_feature_maps_outputs(feature_maps)
        feature_map_heatmap = to_heatmap(feature)
        print(feature_map_heatmap)
        features.append(feature_map_heatmap)
        names.append(name)
    images_to_writer(writer, feature_map_heatmap, 'feature_extractor', names, epoch=0)


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