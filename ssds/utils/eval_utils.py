import torch
import numpy as np

def iou_gt(detect, ground_turths):
    det_size = (detect[2] - detect[0])*(detect[3] - detect[1])
    detect = detect.resize_(1,4)
    iou = []
    ioa = []

    for gt in ground_turths:
        # print(ground_turths)
        # print('gt', gt)
        # print(detect)
        
        gt = gt.resize_(1,4)
        # print('gt', gt)
        gt_size = (gt[0][2] - gt[0][0])*(gt[0][3] - gt[0][1])

        inter_max = torch.max(detect, gt)
        inter_min = torch.min(detect, gt)
        # print(inter_max)
        # print(inter_min)
        inter_size = max(inter_min[0][2] - inter_max[0][0], 0.) * max(inter_min[0][3] - inter_max[0][1], 0.)

        _iou = inter_size / (det_size + gt_size - inter_size)
        _ioa = inter_size / gt_size

        iou.append(_iou)
        ioa.append(_ioa)

        # print(inter_size)
        # print(det_size, gt_size)
        # print(iou)
    return iou, ioa
    

## TODO: currently, it is super time cost. any ideas to improve it?
# def cal_tp_fp(detects, ground_turths, label, score, npos, iou_threshold=0.5, conf_threshold=0.01):
#     '''
#     '''
#     for det, gt in zip(detects, ground_turths):
#         for i, det_c in enumerate(det):            
#             gt_c = [_gt[:4].data.resize_(1,4) for _gt in gt if int(_gt[4]) == i+1]  # only 20 in det
#             if len(det_c) == 0:cls_dets
#                 npos[i] += len(gt_c)
#                 continue
#             # print(det_c)
#             # assert(False)
#             iou_c = []
#             ioa_c = []
#             score_c = []
#             num=0
#             for det_c_n in det_c:
#                 if len(gt_c) > 0:
#                     _iou, _ioa = iou_gt(det_c_n[:4], gt_c)
#                     iou_c.append(_iou)
#                     ioa_c.append(_ioa)
#                 score_c.append(det_c_n[4])
#                 num+=1
            
#             # No detection 
#             if len(iou_c) == 0:
#                 npos[i] += len(gt_c)
#                 continue

#             labels_c = [0] * len(score_c)
#             # TODO: currently ignore the difficulty & ignore the group of boxes.
#             # Tp-fp evaluation for non-group of boxes (if any).
#             if len(gt_c) > 0:
#                 # groundtruth_nongroup_of_is_difficult_list = groundtruth_is_difficult_list[
#                 #     ~groundtruth_is_group_of_list]
#                 max_overlap_gt_ids = np.argmax(np.array(iou_c), axis=1)
#                 is_gt_box_detected = np.zeros(len(gt_c), dtype=bool)
#                 for iters in range(len(labels_c)):
#                     gt_id = max_overlap_gt_ids[iters]
#                     if iou_c[iters][gt_id] >= iou_threshold:
#                         # if not groundtruth_nongroup_of_is_difficult_list[gt_id]:
#                         if not is_gt_box_detected[gt_id]:
#                             labels_c[iters] = 1
#                             is_gt_box_detected[gt_id] = True
#                         # else:
#                         #     is_matched_to_difficult_box[i] = True

#             # append to the global label, score
#             npos[i] += len(gt_c)
#             label[i].extend(labels_c)
#             score[i].extend(score_c)
        
#     return label, score, npos

def cal_tp_fp(detects, ground_turths, label, score, npos, gt_label, iou_threshold=0.5, conf_threshold=0.01):
    '''
    '''
    for det, gt in zip(detects, ground_turths):
        for i, det_c in enumerate(det):            
            gt_c = [_gt[:4].data.resize_(1,4) for _gt in gt if int(_gt[4]) == i] 
            iou_c = []
            ioa_c = []
            score_c = []
            # num=0
            for det_c_n in det_c:
                if det_c_n[0] < conf_threshold:
                    break
                if len(gt_c) > 0:
                    _iou, _ioa = iou_gt(det_c_n[1:], gt_c)
                    iou_c.append(_iou)
                    ioa_c.append(_ioa)
                score_c.append(det_c_n[0])
            # while det_c[num,0] > conf_threshold:
            #     if len(gt_c) > 0:
            #         _iou, _ioa = iou_gt(det_c[num,1:], gt_c)
            #         iou_c.append(_iou)
            #         ioa_c.append(_ioa)
            #     score_c.append(det_c[num, 0])
            #     num+=1
            
            # No detection 
            if len(iou_c) == 0:
                npos[i] += len(gt_c)
                if len(gt_c) > 0:
                    is_gt_box_detected = np.zeros(len(gt_c), dtype=bool)
                    gt_label[i] += is_gt_box_detected.tolist()
                continue

            labels_c = [0] * len(score_c)
            # TODO: currently ignore the difficulty & ignore the group of boxes.
            # Tp-fp evaluation for non-group of boxes (if any).
            if len(gt_c) > 0:
                # groundtruth_nongroup_of_is_difficult_list = groundtruth_is_difficult_list[
                #     ~groundtruth_is_group_of_list]
                max_overlap_gt_ids = np.argmax(np.array(iou_c), axis=1)
                is_gt_box_detected = np.zeros(len(gt_c), dtype=bool)
                for iters in range(len(labels_c)):
                    gt_id = max_overlap_gt_ids[iters]
                    if iou_c[iters][gt_id] >= iou_threshold:
                        # if not groundtruth_nongroup_of_is_difficult_list[gt_id]:
                        if not is_gt_box_detected[gt_id]:
                            labels_c[iters] = 1
                            is_gt_box_detected[gt_id] = True
                        # else:
                        #     is_matched_to_difficult_box[i] = True

            # append to the global label, score
            npos[i] += len(gt_c)
            label[i] += labels_c
            score[i] += score_c
            gt_label[i] += is_gt_box_detected.tolist()
        
    return label, score, npos, gt_label


def cal_size(detects, ground_turths, size):
    for det, gt in zip(detects, ground_turths):
        for i, det_c in enumerate(det):  
            gt_c = [_gt[:4].data.resize_(1,4) for _gt in gt if int(_gt[4]) == i] 
            if len(gt_c) == 0:
                continue
            gt_size_c = [ [(_gt[0][2] - _gt[0][0]), (_gt[0][3] - _gt[0][1])] for _gt in gt_c ]
            # scale_c = [ min(_size) for _size in gt_size_c ]
            size[i] += gt_size_c
    return size

# def get_correct_detection(detects, ground_turths, iou_threshold=0.5, conf_threshold=0.01):
#     detected = list()
#     for det, gt in zip(detects, ground_turths):
#         for i, det_c in enumerate(det):            
#             gt_c = [_gt[:4].data.resize_(1,4) for _gt in gt if int(_gt[4]) == i] 
#             iou_c = []
#             ioa_c = []
#             detected_c = []
#             # num=0
#             for det_c_n in det_c:
#                 if det_c_n[0] < conf_threshold:
#                     break
#                 if len(gt_c) > 0:
#                     _iou, _ioa = iou_gt(det_c_n[1:], gt_c)
#                     if _iou > iou_threshold:
#                         iou_c.append(_iou)
#                         ioa_c.append(_ioa)
#                 detected_c.append(det_c_n[1:])

#             if len(iou_c) == 0:
#                 npos[i] += len(gt_c)
#                 continue

#             labels_c = [0] * len(detected_c)
#             # TODO: currently ignore the difficulty & ignore the group of boxes.
#             # Tp-fp evaluation for non-group of boxes (if any).
#             if len(gt_c) > 0:
#                 # groundtruth_nongroup_of_is_difficult_list = groundtruth_is_difficult_list[
#                 #     ~groundtruth_is_group_of_list]
#                 max_overlap_gt_ids = np.argmax(np.array(iou_c), axis=1)
#                 is_gt_box_detected = np.zeros(len(gt_c), dtype=bool)
#                 for iters in range(len(labels_c)):
#                     gt_id = max_overlap_gt_ids[iters]
#                     if iou_c[iters][gt_id] >= iou_threshold:
#                         # if not groundtruth_nongroup_of_is_difficult_list[gt_id]:
#                         if not is_gt_box_detected[gt_id]:
#                             labels_c[iters] = 1
#                             is_gt_box_detected[gt_id] = True
#             detected_c = detected_c[labels_c]
#             detected.append(detected_c)
        
#     return detected


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

        rec = tp.astype(float) / float(npos)
        prec = tp.astype(float) / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap += [compute_average_precision(prec, rec)]
        recall+=[rec]
        precision+=[prec]
    mAP = np.nanmean(ap)
    return precision, recall, mAP


# def voc_ap(rec, prec, use_07_metric=False):
#     """ ap = voc_ap(rec, prec, [use_07_metric])
#     Compute VOC AP given precision and recall.
#     If use_07_metric is true, uses the
#     VOC 07 11 point method (default:False).
#     """
#     if use_07_metric:
#         # 11 point metric
#         ap = 0.
#         for t in np.arange(0., 1.1, 0.1):
#             if np.sum(rec >= t) == 0:
#                 p = 0
#             else:
#                 p = np.max(prec[rec >= t])
#             ap = ap + p / 11.
#     else:
#         # correct AP calculation
#         # first append sentinel values at the end
#         mrec = np.concatenate(([0.], rec, [1.]))
#         mpre = np.concatenate(([0.], prec, [0.]))

#         # compute the precision envelope
#         for i in range(mpre.size - 1, 0, -1):
#             mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

#         # to calculate area under PR curve, look for points
#         # where X axis (recall) changes value
#         i = np.where(mrec[1:] != mrec[:-1])[0]

#         # and sum (\Delta recall) * prec
#         ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
#     return ap

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