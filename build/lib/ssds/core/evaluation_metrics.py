import torch
import numpy as np


class MeanAveragePrecision(object):
    def __init__(self, num_classes, conf_threshold, iou_threshold):
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.score, self.detect_ismatched, self.target_ismatched = [
            [[] for _ in range(self.num_classes)] for _ in range(3)
        ]
        self.npos = [0 for _ in range(self.num_classes)]

    def __call__(self, detections, targets):
        def matrix_iou(a, b):
            """
            return iou of a and b, numpy version for data augenmentation
            """
            lt = torch.max(a[:, None, :2], b[:, :2])
            rb = torch.min(a[:, None, 2:], b[:, 2:])

            area_i = torch.prod(rb - lt, dim=2) * (lt < rb).all(dim=2)
            area_a = torch.prod(a[:, 2:] - a[:, :2], dim=1)
            area_b = torch.prod(b[:, 2:] - b[:, :2], dim=1)
            return area_i / (area_a[:, None] + area_b - area_i)

        for out_score, out_box, out_class, target in zip(*detections, targets):
            out_class = out_class[out_score > self.conf_threshold]
            out_box = out_box[out_score > self.conf_threshold]
            out_score = out_score[out_score > self.conf_threshold]
            for c in range(self.num_classes):
                target_c = target[target[:, 4] == c]
                out_score_c = out_score[out_class == c]
                out_box_c = out_box[out_class == c]
                if len(out_score_c) == 0:
                    self.npos[c] += len(target_c)
                    self.target_ismatched[c] += np.zeros(
                        len(target_c), dtype=bool
                    ).tolist()
                    continue
                if len(target_c) == 0:
                    self.score[c] += out_score_c.cpu().tolist()
                    self.detect_ismatched[c] += np.zeros(
                        len(out_score_c), dtype=bool
                    ).tolist()
                    continue
                iou_c = matrix_iou(out_box_c, target_c[:, :4])
                max_overlap_tids = torch.argmax(iou_c, dim=1)
                is_box_detected = np.zeros(len(target_c), dtype=bool)
                lable_c = np.zeros(len(out_score_c), dtype=bool)
                for i in range(len(max_overlap_tids)):
                    tid = max_overlap_tids[i]
                    if iou_c[i][tid] >= self.iou_threshold and not is_box_detected[tid]:
                        is_box_detected[tid] = True
                        lable_c[i] = True
                self.npos[c] += len(target_c)
                self.detect_ismatched[c] += lable_c.tolist()
                self.score[c] += out_score_c.cpu().tolist()
                self.target_ismatched[c] += is_box_detected.tolist()
        return

    def get_results(self):
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

            if not isinstance(precision, np.ndarray) or not isinstance(
                recall, np.ndarray
            ):
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
                (recall[indices] - recall[indices - 1]) * precision[indices]
            )
            return average_precision

        recall, precision, ap = [], [], []
        for labels_c, scores_c, npos_c in zip(
            self.detect_ismatched, self.score, self.npos
        ):
            # to avoid missing ground truth in that class
            if npos_c == 0:
                ap += [np.NAN]
                recall += [[0], [1]]
                precision += [[0], [0]]
                continue

            sorted_indices = np.argsort(scores_c)
            sorted_indices = sorted_indices[::-1]
            labels_c = np.array(labels_c).astype(int)
            true_positive_labels = labels_c[sorted_indices]
            false_positive_labels = 1 - true_positive_labels
            tp = np.cumsum(true_positive_labels)
            fp = np.cumsum(false_positive_labels)

            rec = tp.astype(float) / float(npos_c)
            prec = tp.astype(float) / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap += [compute_average_precision(prec, rec)]
            recall += [rec]
            precision += [prec]
        mAP = np.nanmean(ap)
        return mAP, (precision, recall, ap)
