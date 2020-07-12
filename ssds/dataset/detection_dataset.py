import copy
import sys
import pickle
import glob

import cv2
import numpy as np
from PIL import Image
import io

import torch
import torch.utils.data as data

from . import transforms as preprocess

class DetectionDataset(data.Dataset):
    '''The detection 2d dataset. Used to get the images
    '''
    def __init__(self, cfg, is_train, transform=None):
        # super(DetectionDataset, self).__init__()
        self.is_train = is_train

        self.image_size = cfg.IMAGE_SIZE
        # self.num_classes = cfg.NUM_CLASSES
        # self.classes_names = cfg.CLASSES_NAME
        self.preproc_param = cfg.PREPROC
        self.using_pickle = cfg.PICKLE
        self.transform = transform

        self.db = []
        self.img_db = []
        self._init_transform()

    def _init_transform(self):
        if self.is_train:
            self.transform = preprocess.Compose([
                preprocess.ConvertFromInts(),
                preprocess.ToAbsoluteCoords(),
                preprocess.RandomSampleCrop(scale=self.preproc_param.CROP_SCALE, 
                                            num_attempts=self.preproc_param.CROP_ATTEMPTS),
                preprocess.RandomMirror(),
                # preprocess.PhotometricDistort(hue_delta=self.preproc_param.HUE_DELTA,
                #                               bri_delta=self.preproc_param.BRI_DELTA, 
                #                               contrast_range=self.preproc_param.CONTRAST_RANGE, 
                #                               saturation_range=self.preproc_param.SATURATION_RANGE),
                preprocess.Expand(mean=self.preproc_param.MEAN, 
                                  max_expand_ratio=self.preproc_param.MAX_EXPAND_RATIO),
                preprocess.ToPercentCoords(),
                preprocess.Resize(tuple(self.image_size)),
                preprocess.ToAbsoluteCoords(),
                preprocess.ToTensor(),
                # preprocess.ToGPU(),
                preprocess.Normalize(mean=self.preproc_param.MEAN, std=self.preproc_param.STD),
                preprocess.ToXYWH(),
            ])
        else:
            self.transform = preprocess.Compose([
                preprocess.ConvertFromInts(),
                preprocess.Resize(tuple(self.image_size)),
                preprocess.ToAbsoluteCoords(),
                preprocess.ToTensor(),
                # preprocess.ToGPU(),
                preprocess.Normalize(mean=self.preproc_param.MEAN, std=self.preproc_param.STD),
                preprocess.ToXYWH(),
            ])

    def _get_db(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        '''
        Args:
            index for db, 
            db[index] = {
                'image': 'Absolute Path',
                'boxes': np.ndarray
                'labels': np.adarray
            }

        Returns:
            'image': torch(c,h,w),
            'target': np.ndarray(n,5)
                    0~4 is the bounding box in AbsoluteCoords with format x,y,w,h
                    5 is the bounding box label
        '''
        db_rec = copy.deepcopy(self.db[index])
        
        # read the images
        if self.using_pickle:
            # decode image
            encoded_image = copy.deepcopy(self.img_db[index])
            image = Image.open(io.BytesIO(encoded_image))
            image = np.array(image) 
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        else:
            image_file = db_rec['image']
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
            if image is None:
                raise ValueError('Fail to read {}'.format(image_file))

        boxes = db_rec['boxes']
        labels = db_rec['labels']

        # preprocess
        image, boxes, labels = self.transform(image, boxes, labels)
        return image, np.concatenate((boxes, labels[:,None]),axis=1)

    def reorder_data(self, db, cfg_joints_name, ds_joints_name):
        ''' reorder the db based on the cfg_joints_name
        '''
        order = []
        for cfg_name in cfg_joints_name:
            if cfg_name in ds_joints_name:
                order.append(ds_joints_name.index(cfg_name))
            else:
                order.append(-1)
        order = np.array(order)

        raise NotImplementedError
        return db

    def saving_pickle(self, pickle_path):
        img_db = []
        for idx, db_rec in enumerate(self.db):
            sys.stdout.write('\rLoading Image: {}/{}'.format(idx, len(self.db)))
            sys.stdout.flush()
            # load bytes from file
            with open(db_rec['image'], 'rb') as f:
                img_db.append(f.read())

        # serialize
        sys.stdout.write('\rSaving img_db ({}) to {}\n'.format(len(self.db), pickle_path))
        with open(pickle_path, 'wb') as handle:
            return pickle.dump(img_db, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def loading_pickle(self, pickle_path):
        sys.stdout.write('\rLoading Pickle from {}\n'.format(pickle_path))
        with open(pickle_path, 'rb') as handle:
            return pickle.load(handle)