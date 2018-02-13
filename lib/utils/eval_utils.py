import torch
import numpy as np

# def area(boxlist):
#   """Computes area of boxes.

#   Args:
#     boxlist: BoxList holding N boxes

#   Returns:
#     a numpy array with shape [N*1] representing box areas
#   """
#   y_min, x_min, y_max, x_max = boxlist.get_coordinates()
#   return (y_max - y_min) * (x_max - x_min)

# def intersection(boxes1, boxes2):
#   """Compute pairwise intersection areas between boxes.

#   Args:
#     boxes1: a numpy array with shape [N, 4] holding N boxes
#     boxes2: a numpy array with shape [M, 4] holding M boxes

#   Returns:
#     a numpy array with shape [N*M] representing pairwise intersection area
#   """
#   [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
#   [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

#   all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
#   all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
#   intersect_heights = np.maximum(
#       np.zeros(all_pairs_max_ymin.shape),
#       all_pairs_min_ymax - all_pairs_max_ymin)
#   all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
#   all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
#   intersect_widths = np.maximum(
#       np.zeros(all_pairs_max_xmin.shape),
#       all_pairs_min_xmax - all_pairs_max_xmin)
#   return intersect_heights * intersect_widths

# def iou(boxes1, boxes2):
#   """Computes pairwise intersection-over-union between box collections.

#   Args:
#     boxes1: a numpy array with shape [N, 4] holding N boxes.
#     boxes2: a numpy array with shape [M, 4] holding N boxes.

#   Returns:
#     a numpy array with shape [N, M] representing pairwise iou scores.
#   """
#   intersect = intersection(boxes1, boxes2)
#   area1 = area(boxes1)
#   area2 = area(boxes2)
#   union = np.expand_dims(area1, axis=1) + np.expand_dims(
#       area2, axis=0) - intersect
#   return intersect / union


# def ioa(boxes1, boxes2):
#   """Computes pairwise intersection-over-area between box collections.

#   Intersection-over-area (ioa) between two boxes box1 and box2 is defined as
#   their intersection area over box2's area. Note that ioa is not symmetric,
#   that is, IOA(box1, box2) != IOA(box2, box1).

#   Args:
#     boxes1: a numpy array with shape [N, 4] holding N boxes.
#     boxes2: a numpy array with shape [M, 4] holding N boxes.

#   Returns:
#     a numpy array with shape [N, M] representing pairwise ioa scores.
#   """
#   intersect = intersection(boxes1, boxes2)
#   areas = np.expand_dims(area(boxes2), axis=0)
#   return intersect / areas

# class BoxList(object):
#   """Box collection.

#   BoxList represents a list of bounding boxes as numpy array, where each
#   bounding box is represented as a row of 4 numbers,
#   [y_min, x_min, y_max, x_max].  It is assumed that all bounding boxes within a
#   given list correspond to a single image.

#   Optionally, users can add additional related fields (such as
#   objectness/classification scores).
#   """

#   def __init__(self, data):
#     """Constructs box collection.

#     Args:
#       data: a numpy array of shape [N, 4] representing box coordinates

#     Raises:
#       ValueError: if bbox data is not a numpy array
#       ValueError: if invalid dimensions for bbox data
#     """
#     if not isinstance(data, np.ndarray):
#       raise ValueError('data must be a numpy array.')
#     if len(data.shape) != 2 or data.shape[1] != 4:
#       raise ValueError('Invalid dimensions for box data.')
#     if data.dtype != np.float32 and data.dtype != np.float64:
#       raise ValueError('Invalid data type for box data: float is required.')
#     if not self._is_valid_boxes(data):
#       raise ValueError('Invalid box data. data must be a numpy array of '
#                        'N*[y_min, x_min, y_max, x_max]')
#     self.data = {'boxes': data}

#   def num_boxes(self):
#     """Return number of boxes held in collections."""
#     return self.data['boxes'].shape[0]

#   def get_extra_fields(self):
#     """Return all non-box fields."""
#     return [k for k in self.data.keys() if k != 'boxes']

#   def has_field(self, field):
#     return field in self.data

#   def add_field(self, field, field_data):
#     """Add data to a specified field.

#     Args:
#       field: a string parameter used to speficy a related field to be accessed.
#       field_data: a numpy array of [N, ...] representing the data associated
#           with the field.
#     Raises:
#       ValueError: if the field is already exist or the dimension of the field
#           data does not matches the number of boxes.
#     """
#     if self.has_field(field):
#       raise ValueError('Field ' + field + 'already exists')
#     if len(field_data.shape) < 1 or field_data.shape[0] != self.num_boxes():
#       raise ValueError('Invalid dimensions for field data')
#     self.data[field] = field_data

#   def get(self):
#     """Convenience function for accesssing box coordinates.

#     Returns:
#       a numpy array of shape [N, 4] representing box corners
#     """
#     return self.get_field('boxes')

#   def get_field(self, field):
#     """Accesses data associated with the specified field in the box collection.

#     Args:
#       field: a string parameter used to speficy a related field to be accessed.

#     Returns:
#       a numpy 1-d array representing data of an associated field

#     Raises:
#       ValueError: if invalid field
#     """
#     if not self.has_field(field):
#       raise ValueError('field {} does not exist'.format(field))
#     return self.data[field]

#   def get_coordinates(self):
#     """Get corner coordinates of boxes.

#     Returns:
#      a list of 4 1-d numpy arrays [y_min, x_min, y_max, x_max]
#     """
#     box_coordinates = self.get()
#     y_min = box_coordinates[:, 0]
#     x_min = box_coordinates[:, 1]
#     y_max = box_coordinates[:, 2]
#     x_max = box_coordinates[:, 3]
#     return [y_min, x_min, y_max, x_max]

#   def _is_valid_boxes(self, data):
#     """Check whether data fullfills the format of N*[ymin, xmin, ymax, xmin].

#     Args:
#       data: a numpy array of shape [N, 4] representing box coordinates

#     Returns:
#       a boolean indicating whether all ymax of boxes are equal or greater than
#           ymin, and all xmax of boxes are equal or greater than xmin.
#     """
#     if data.shape[0] > 0:
#       for i in range(data.shape[0]):
#         if data[i, 0] > data[i, 2] or data[i, 1] > data[i, 3]:
#           return False
#     return True

# def _compute_tp_fp_for_single_class(
#     detected_boxes, detected_scores, groundtruth_boxes,
#     groundtruth_is_difficult_list, groundtruth_is_group_of_list):
#     """Labels boxes detected with the same class from the same image as tp/fp.

#     Args:
#         detected_boxes: A numpy array of shape [N, 4] representing detected box
#             coordinates
#         detected_scores: A 1-d numpy array of length N representing classification
#             score
#         groundtruth_boxes: A numpy array of shape [M, 4] representing ground truth
#             box coordinates
#         groundtruth_is_difficult_list: A boolean numpy array of length M denoting
#             whether a ground truth box is a difficult instance or not. If a
#             groundtruth box is difficult, every detection matching this box
#             is ignored.
#         groundtruth_is_group_of_list: A boolean numpy array of length M denoting
#             whether a ground truth box has group-of tag. If a groundtruth box
#             is group-of box, every detection matching this box is ignored.

#     Returns:
#         Two arrays of the same size, containing all boxes that were evaluated as
#         being true positives or false positives; if a box matched to a difficult
#         box or to a group-of box, it is ignored.

#         scores: A numpy array representing the detection scores.
#         tp_fp_labels: a boolean numpy array indicating whether a detection is a
#             true positive.

#     """
#     if detected_boxes.size == 0:
#         return np.array([], dtype=float), np.array([], dtype=bool)
#     detected_boxlist = BoxList(detected_boxes)
#     detected_boxlist.add_field('scores', detected_scores)
#     detected_boxlist = np_box_list_ops.non_max_suppression(
#         detected_boxlist, self.nms_max_output_boxes, self.nms_iou_threshold)

#     scores = detected_boxlist.get_field('scores')

#     if groundtruth_boxes.size == 0:
#         return scores, np.zeros(detected_boxlist.num_boxes(), dtype=bool)

#     tp_fp_labels = np.zeros(detected_boxlist.num_boxes(), dtype=bool)
#     is_matched_to_difficult_box = np.zeros(
#         detected_boxlist.num_boxes(), dtype=bool)
#     is_matched_to_group_of_box = np.zeros(
#         detected_boxlist.num_boxes(), dtype=bool)

#     # The evaluation is done in two stages:
#     # 1. All detections are matched to non group-of boxes; true positives are
#     #    determined and detections matched to difficult boxes are ignored.
#     # 2. Detections that are determined as false positives are matched against
#     #    group-of boxes and ignored if matched.

#     # Tp-fp evaluation for non-group of boxes (if any).
#     gt_non_group_of_boxlist = np_box_list.BoxList(
#         groundtruth_boxes[~groundtruth_is_group_of_list, :])
#     if gt_non_group_of_boxlist.num_boxes() > 0:
#         groundtruth_nongroup_of_is_difficult_list = groundtruth_is_difficult_list[
#             ~groundtruth_is_group_of_list]
#         iou = np_box_list_ops.iou(detected_boxlist, gt_non_group_of_boxlist)
#         max_overlap_gt_ids = np.argmax(iou, axis=1)
#         is_gt_box_detected = np.zeros(
#             gt_non_group_of_boxlist.num_boxes(), dtype=bool)
#         for i in range(detected_boxlist.num_boxes()):
#             gt_id = max_overlap_gt_ids[i]
#         if iou[i, gt_id] >= self.matching_iou_threshold:
#             if not groundtruth_nongroup_of_is_difficult_list[gt_id]:
#                 if not is_gt_box_detected[gt_id]:
#                     tp_fp_labels[i] = True
#                     is_gt_box_detected[gt_id] = True
#                 else:
#                     is_matched_to_difficult_box[i] = True

#     # Tp-fp evaluation for group of boxes.
#     gt_group_of_boxlist = BoxList(
#         groundtruth_boxes[groundtruth_is_group_of_list, :])
#     if gt_group_of_boxlist.num_boxes() > 0:
#         ioa = ioa(gt_group_of_boxlist.get(), detected_boxlist.get())
#         max_overlap_group_of_gt = np.max(ioa, axis=0)
#         for i in range(detected_boxlist.num_boxes()):
#             if (not tp_fp_labels[i] and not is_matched_to_difficult_box[i] and
#                 max_overlap_group_of_gt[i] >= self.matching_iou_threshold):
#                 is_matched_to_group_of_box[i] = True

#     return scores[~is_matched_to_difficult_box
#                     & ~is_matched_to_group_of_box], tp_fp_labels[
#                         ~is_matched_to_difficult_box
#                         & ~is_matched_to_group_of_box]


def iou_gt(detect, ground_turths, iou_threshold=0.5):
    det_size = (detect[2] - detect[0])*(detect[3] - detect[1])
    detect = detect.resize_(1,4)
    for gt in ground_turths:
        # print(ground_turths)
        # print('gt', gt)
        # print(detect)
        
        gt = gt.data.resize_(1,4)
        gt_size = (gt[0][2] - gt[0][0])*(gt[0][3] - gt[0][1])

        inter_max = torch.max(detect, gt)
        inter_min = torch.min(detect, gt)
        # print(inter_max)
        # print(inter_min)
        inter_size = max(inter_min[0][2] - inter_max[0][0], 0.) * max(inter_min[0][3] - inter_max[0][1], 0.)

        iou = inter_size / (det_size + gt_size - inter_size)
        # print(inter_size)
        # print(det_size, gt_size)
        # print(iou)
        if iou > iou_threshold:
            return True
        # assert(False)
    return False

def cal_tp_fp(detects, ground_turths, label, score, npos, iou_threshold=0.5, conf_threshold=0.1):

    # print(detects[0])
    # print(ground_turths)

    for det, gt in zip(detects, ground_turths):
        # print(det)
        for i, det_c in enumerate(det):
            gt_c = [_gt[:4] for _gt in gt if int(_gt[4]) == i] #should from class 1
            # if len(gt_c) !=0:
            #     print(det_c)
            #     print(gt_c)
            #     assert(False)
            num=0
            npos[i] += len(gt_c)
            while det_c[num,0] > conf_threshold:
                if iou_gt(det_c[num,1:], gt_c):
                    label[i].append(1)
                else:
                    label[i].append(0)
                score[i].append(det_c[num, 0])
                num+=1

        # npos += len(gt.view(-1,5))
    # print(tp)
    # print(fp)
    return label, score, npos


def cal_pr(_label, _score, _npos):
    recall = []
    precision = []
    ap = []
    for labels, scores, npos in zip(_label[1:], _score[1:], _npos[1:]):
        sorted_indices = np.argsort(scores)
        sorted_indices = sorted_indices[::-1]
        labels = np.array(labels).astype(int)
        true_positive_labels = labels[sorted_indices]
        false_positive_labels = 1 - true_positive_labels
        tp = np.cumsum(true_positive_labels)
        fp = np.cumsum(false_positive_labels)

        print(scores)
        print(true_positive_labels)
        print(tp)
        print(fp)
        print(npos)
        # assert(False)

        rec = tp.astype(float) / float(npos)
        print(rec)
        prec = tp.astype(float) / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap += [compute_average_precision(prec, rec)]
        recall+=[rec]
        precision+=[prec]
    #print(ap)
    mAP = np.nanmean(ap)
    # print(precision)
    # print(recall)
    # assert(False)
    return precision, recall, mAP


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def compute_average_precision(precision, recall):
  """Compute Average Precision according to the definition in VOCdevkit.

  Precision is modified to ensure that it does not decrease as recall
  decrease.

  Args:
    precision: A float [N, 1] numpy array of precisions
    recall: A float [N, 1] numpy array of recalls

  Raises:
    ValueError: if the input is not of the correct format

  Returns:
    average_precison: The area under the precision recall curve. NaN if
      precision and recall are None.

  """
  if precision is None:
    if recall is not None:
      raise ValueError("If precision is None, recall must also be None")
    return np.NAN

  if not isinstance(precision, np.ndarray) or not isinstance(recall,
                                                             np.ndarray):
    raise ValueError("precision and recall must be numpy array")
  if precision.dtype != np.float or recall.dtype != np.float:
    raise ValueError("input must be float numpy array.")
  if len(precision) != len(recall):
    raise ValueError("precision and recall must be of the same size.")
  if not precision.size:
    return 0.0
  if np.amin(precision) < 0 or np.amax(precision) > 1:
    raise ValueError("Precision must be in the range of [0, 1].")
  if np.amin(recall) < 0 or np.amax(recall) > 1:
    raise ValueError("recall must be in the range of [0, 1].")
  if not all(recall[i] <= recall[i + 1] for i in range(len(recall) - 1)):
    raise ValueError("recall must be a non-decreasing array")

  recall = np.concatenate([[0], recall, [1]])
  precision = np.concatenate([[0], precision, [0]])

  # Preprocess precision to be a non-decreasing array
  for i in range(len(precision) - 2, -1, -1):
    precision[i] = np.maximum(precision[i], precision[i + 1])

  indices = np.where(recall[1:] != recall[:-1])[0] + 1
  average_precision = np.sum(
      (recall[indices] - recall[indices - 1]) * precision[indices])
  return average_precision