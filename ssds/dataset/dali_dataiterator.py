import torch
import math
import ctypes

import nvidia.dali.ops as ops
import nvidia.dali.types as types


class DaliPipeline(object):
    r""" The data pipeline for the Dali dataset
    """

    def __init__(self, target_size, preproc_param, training=False):
        self.training = training
        mean = preproc_param.MEAN
        std = preproc_param.STD
        bri_delta = preproc_param.BRI_DELTA
        hue_delta = preproc_param.HUE_DELTA
        max_expand_ratio = preproc_param.MAX_EXPAND_RATIO
        contrast_range = preproc_param.CONTRAST_RANGE
        saturation_range = preproc_param.SATURATION_RANGE
        crop_aspect_ratio = preproc_param.CROP_ASPECT_RATIO
        crop_scale = preproc_param.CROP_SCALE
        crop_attempts = preproc_param.CROP_ATTEMPTS

        # decoder
        self.decode_train = ops.ImageDecoderSlice(device="mixed", output_type=types.RGB)
        self.decode_infer = ops.ImageDecoder(device="mixed", output_type=types.RGB)

        # ssd crop
        self.bbox_crop = ops.RandomBBoxCrop(
            device="cpu",
            bbox_layout="xyXY",
            scaling=crop_scale,
            aspect_ratio=crop_aspect_ratio,
            allow_no_crop=True,
            thresholds=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
            num_attempts=crop_attempts,
        )

        # color twist
        self.uniform_con = ops.Uniform(range=contrast_range)
        self.uniform_bri = ops.Uniform(
            range=(1.0 - bri_delta / 256.0, 1.0 + bri_delta / 256.0)
        )
        self.uniform_sat = ops.Uniform(range=saturation_range)
        self.uniform_hue = ops.Uniform(range=(-hue_delta, hue_delta))
        self.hsv = ops.Hsv(device="gpu")
        self.contrast = ops.BrightnessContrast(device="gpu")

        # hflip
        self.bbox_flip = ops.BbFlip(device="cpu", ltrb=True)
        self.img_flip = ops.Flip(device="gpu")
        self.coin_flip = ops.CoinFlip(probability=0.5)

        # past
        self.paste_pos = ops.Uniform(range=(0, 1))
        self.paste_ratio = ops.Uniform(range=(1, max_expand_ratio))
        self.paste = ops.Paste(device="gpu", fill_value=mean)
        self.bbox_paste = ops.BBoxPaste(device="cpu", ltrb=True)

        # resize and normalize
        self.resize = ops.Resize(
            device="gpu",
            interp_type=types.DALIInterpType.INTERP_CUBIC,
            resize_x=target_size[0],
            resize_y=target_size[1],
            save_attrs=True,
        )
        self.normalize = ops.CropMirrorNormalize(device="gpu", mean=mean, std=std)

    def predefined_graph(self, images, bboxes, labels):
        if self.training:
            # crop
            crop_begin, crop_size, bboxes, labels = self.bbox_crop(bboxes, labels)
            images = self.decode_train(images, crop_begin, crop_size)

            # color twist
            images = self.hsv(
                images, hue=self.uniform_hue(), saturation=self.uniform_sat()
            )
            images = self.contrast(
                images, brightness=self.uniform_bri(), contrast=self.uniform_con()
            )

            # hflip
            flip = self.coin_flip()
            bboxes = self.bbox_flip(bboxes, horizontal=flip)
            images = self.img_flip(images, horizontal=flip)

            # past
            ratio = self.paste_ratio()
            px = self.paste_pos()
            py = self.paste_pos()
            images = self.paste(images.gpu(), paste_x=px, paste_y=py, ratio=ratio)
            bboxes = self.bbox_paste(bboxes, paste_x=px, paste_y=py, ratio=ratio)
        else:
            images = self.decode_infer(images)

        images, attrs = self.resize(images)
        images = self.normalize(images)

        return images, bboxes, labels


class DaliDataset(object):
    r""" Data loader for data parallel using Dali
    """

    def __init__(self, cfg, dataset_dir, image_sets, batch_size, training=False):

        self.training = training
        self.batch_size = batch_size
        self.target_size = cfg.IMAGE_SIZE
        self.preproc_param = cfg.PREPROC

        self.device_ids = (
            torch.cuda.current_device() if len(cfg.DEVICE_ID) != 1 else cfg.DEVICE_ID[0]
        )  # ",".join([str(d) for d in device_ids])
        self.num_shards = max(len(cfg.DEVICE_ID), 1)
        self.num_threads = cfg.NUM_WORKERS

        self.pipeline_args = {
            "target_size": self.target_size,
            "num_threads": self.num_threads,
            "num_shards": self.num_shards,
            "batch_size": self.batch_size,
            "training": self.training,
            "device_ids": self.device_ids,
            "preproc_param": self.preproc_param,
        }

    def __repr__(self):
        return "\n".join(
            [
                "    loader: dali"
                "    length: {}"
                "    target_size: {}".format(self.__len__(), self.target_size),
            ]
        )

    def __len__(self):
        return math.ceil(len(self.pipe) // self.num_shards / self.batch_size)

    def __iter__(self):
        for _ in range(self.__len__()):
            data, num_detections = [], []
            dali_data, dali_boxes, dali_labels = self.pipe.run()

            for l in range(len(dali_boxes)):
                num_detections.append(dali_boxes.at(l).shape[0])

            torch_targets = -1 * torch.ones(
                [len(dali_boxes), max(max(num_detections), 1), 5]
            )

            for batch in range(self.batch_size):
                # Convert dali tensor to pytorch
                dali_tensor = dali_data[batch]
                tensor_shape = dali_tensor.shape()

                datum = torch.zeros(
                    dali_tensor.shape(), dtype=torch.float, device=torch.device("cuda")
                )
                c_type_pointer = ctypes.c_void_p(datum.data_ptr())
                dali_tensor.copy_to_external(c_type_pointer)

                # Rescale boxes
                b_arr = dali_boxes.at(batch)
                num_dets = b_arr.shape[0]
                if num_dets is not 0:
                    torch_bbox = torch.from_numpy(b_arr).float()

                    torch_bbox[:, ::2] *= self.target_size[0]
                    torch_bbox[:, 1::2] *= self.target_size[1]
                    # (l,t,r,b) ->  (x,y,w,h) == (l,r, r-l, b-t)
                    torch_bbox[:, 2] -= torch_bbox[:, 0]
                    torch_bbox[:, 3] -= torch_bbox[:, 1]
                    torch_targets[batch, :num_dets, :4] = torch_bbox  # * ratio

                # Arrange labels in target tensor
                l_arr = dali_labels.at(batch)
                if num_dets is not 0:
                    torch_label = torch.from_numpy(l_arr).float()
                    torch_label -= 1  # Rescale labels to [0,n-1] instead of [1,n]
                    torch_targets[batch, :num_dets, 4] = torch_label.squeeze()

                data.append(datum.unsqueeze(0))

            data = torch.cat(data, dim=0)
            torch_targets = torch_targets.cuda(non_blocking=True)
            yield data, torch_targets

    def reset_size(self, batch_size, target_size):
        r"""
        :meta private:
        """
        raise NotImplementedError()
