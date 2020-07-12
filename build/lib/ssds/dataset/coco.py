import os
import sys
import numpy as np
import pickle
from pycocotools.coco import COCO

from .detection_dataset import DetectionDataset

class COCODataset(object):
    """COCO Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """
    def __init__(self, dataset_dir, image_sets):
        self.dataset_dir   = dataset_dir
        self.cache_path    = os.path.join(dataset_dir, 'cache')
        self.image_sets    = image_sets
        self.img_paths     = []
        self.anno          = []
        self.classes_names = []

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        self._load_anno_files(dataset_dir, image_sets)

    def _load_anno_files(self, dataset_dir, image_sets):
        for coco_name in image_sets:
            annofile = os.path.join(dataset_dir, 'annotations', 'instances_' + coco_name + '.json')
            _COCO = COCO(annofile)
            cats = _COCO.loadCats(_COCO.getCatIds())
            indexes = _COCO.getImgIds()

            self.classes_names = tuple(c['name'] for c in cats)
            self.num_classes = len(self.classes_names)
            self._class_to_ind = dict(zip(self.classes_names, range(self.num_classes)))
            self._class_to_coco_cat_id = dict(zip([c['name'] for c in cats],
                                                  _COCO.getCatIds()))
            self.img_paths.extend(self._load_coco_img_path(coco_name, indexes))
            self.anno.extend(self._load_coco_annotations(coco_name, indexes, _COCO))
            
    def _load_coco_img_path(self, coco_name, indexes):
        cache_file=os.path.join(self.cache_path, coco_name+'_img_path.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                img_path = pickle.load(fid)
            print('{} img path loaded from {}'.format(coco_name,cache_file))
            return img_path

        print('parsing img path for {}'.format(coco_name))
        img_path = [self.image_path_from_index(coco_name, index)
                    for index in indexes]
        with open(cache_file, 'wb') as fid:
            pickle.dump(img_path,fid,pickle.HIGHEST_PROTOCOL)
        print('wrote img path to {}'.format(cache_file))
        return img_path

    def _load_coco_annotations(self, coco_name, indexes, _COCO):
        cache_file=os.path.join(self.cache_path, coco_name+'_gt_db.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt loaded from {}'.format(coco_name,cache_file))
            return roidb

        print('parsing gt for {}'.format(coco_name))
        gt_roidb = [self.annotation_from_index(index, _COCO)
                    for index in indexes]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb,fid,pickle.HIGHEST_PROTOCOL)
        print('wrote gt to {}'.format(cache_file))
        return gt_roidb

    def image_path_from_index(self, name, index):
        """
        Construct an image path from the image's "index" identifier.
        Example image path for index=119993:
          images/train2014/COCO_train2014_000000119993.jpg
        """
        file_name = (str(index).zfill(12) + '.jpg')
        image_path = os.path.join(self.dataset_dir, 'images',
                              name, file_name)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def annotation_from_index(self, index, _COCO, toPercent=True):
        """
        Loads COCO bounding-box instance annotations. Crowd instances are
        handled by marking their overlaps (with all categories) to -1. This
        overlap value means that crowd "instances" are excluded from training.
        Return result with Percent Coords
        """
        im_ann = _COCO.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = _COCO.getAnnIds(imgIds=index, iscrowd=None)
        objs = _COCO.loadAnns(annIds)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)

        # Lookup table to map from COCO category ids to our internal class
        # indices
        coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[name],
                                          self._class_to_ind[name])
                                         for name in self.classes_names])

        res = np.zeros((len(valid_objs), 5), dtype=np.float32)
        for ix, obj in enumerate(valid_objs):
            clss = coco_cat_id_to_class_ind[obj['category_id']]
            res[ix, 0:4] = obj['clean_bbox']
            res[ix, 4] = clss

        if toPercent == True:
            res[:,:4:2] /= width
            res[:,1:4:2] /= height
        return res

class COCODetection(COCODataset, DetectionDataset):
    def __init__(self, cfg, dataset_dir, image_sets, training=False, transform=None): 
        DetectionDataset.__init__(self, cfg, training, transform)
        COCODataset.__init__(self, dataset_dir, image_sets)

        self.db = self._get_db()
        # self.db = self.reorder_data(self.db, self.cfg_joints_name, self.ds_joints_name)

        # loading img db to boost up the speed
        if self.using_pickle:
            pickle_path = os.path.join(dataset_dir, 'pickle', 'img_db_' + '_'.join(image_set) + '.pickle')
            if not os.path.exists(os.path.dirname(pickle_path)):
                os.makedirs(os.path.dirname(pickle_path))
            if not os.path.exists(pickle_path):
                self.saving_pickle(pickle_path)
            self.img_db = self.loading_pickle(pickle_path)

    def _get_db(self):
        gt_db = [{
            'image': img_path,
            'boxes': anno[:,:4],
            'labels': anno[:,4]
        } for img_path, anno in zip(self.img_paths, self.anno)]
        return gt_db