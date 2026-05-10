"""Non-Maximum Suppression utilities for object detection post-processing.

Supports standard NMS and YOLO-specific decoding (bounding box formats).
"""

import numpy as np
from typing import List, Tuple, Optional


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.001,
    max_detections: int = 300,
) -> np.ndarray:
    """Perform Non-Maximum Suppression on bounding boxes.

    Args:
        boxes: Array of shape [N, 4] with [x1, y1, x2, y2] coordinates.
        scores: Array of shape [N] with detection scores.
        iou_threshold: IoU threshold for suppression.
        score_threshold: Minimum score threshold.
        max_detections: Maximum number of detections to return.

    Returns:
        Array of indices of kept detections.
    """
    keep = np.where(scores >= score_threshold)[0]
    if len(keep) == 0:
        return np.array([], dtype=np.int64)

    boxes = boxes[keep]
    scores = scores[keep]

    order = np.argsort(-scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    selected = []
    while len(order) > 0 and len(selected) < max_detections:
        i = order[0]
        selected.append(i)

        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        mask = np.where(iou <= iou_threshold)[0]
        order = order[1:][mask]

    return keep[np.array(selected)]


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [x_center, y_center, width, height] to [x1, y1, x2, y2].

    Args:
        boxes: Array of shape [N, 4] in xywh format.

    Returns:
        Array of shape [N, 4] in xyxy format.
    """
    boxes = boxes.copy()
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    return boxes


def scale_boxes(
    img1_shape: Tuple[int, int],
    boxes: np.ndarray,
    img0_shape: Tuple[int, int],
    ratio_pad: Optional[Tuple[float, Tuple[int, int]]] = None,
) -> np.ndarray:
    """Rescale boxes from model input size to original image size.

    Args:
        img1_shape: Model input shape (height, width).
        boxes: Array of shape [N, 4] in xyxy format.
        img0_shape: Original image shape (height, width).
        ratio_pad: Optional pre-computed (ratio, (pad_h, pad_w)).

    Returns:
        Scaled boxes in xyxy format relative to original image.
    """
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad_h = int((img1_shape[0] - img0_shape[0] * gain) / 2)
        pad_w = int((img1_shape[1] - img0_shape[1] * gain) / 2)
        ratio = gain
    else:
        ratio, (pad_h, pad_w) = ratio_pad

    boxes = boxes.copy()
    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    boxes[:, :4] /= ratio

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img0_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img0_shape[0])

    return boxes


def yolo_decode(
    output: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 100,
    num_classes: int = 80,
) -> List[np.ndarray]:
    """Decode YOLO model output into detections.

    Supports YOLOv5/v8/v11 format: [batch, num_detections, 4 + num_classes]
    where the 4 are [x_center, y_center, width, height].

    Args:
        output: Model output tensor as numpy array [batch, num_detections, 4 + num_classes].
        conf_threshold: Confidence threshold.
        iou_threshold: IoU threshold for NMS.
        max_detections: Maximum detections per image.
        num_classes: Number of classes.

    Returns:
        List of detections per image. Each detection is [N, 6] where
        columns are [x1, y1, x2, y2, confidence, class_id].
    """
    if output.ndim == 2:
        output = output[np.newaxis, :, :]

    batch_detections = []
    for i in range(output.shape[0]):
        pred = output[i]

        boxes = xywh2xyxy(pred[:, :4])

        class_scores = pred[:, 4:]

        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        valid = max_scores >= conf_threshold
        if not np.any(valid):
            batch_detections.append(np.empty((0, 6)))
            continue

        boxes = boxes[valid]
        scores = max_scores[valid]
        class_ids = class_ids[valid]

        all_detections = []
        unique_classes = np.unique(class_ids)

        for cls in unique_classes:
            cls_mask = class_ids == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]

            indices = nms(cls_boxes, cls_scores, iou_threshold=iou_threshold)

            for idx in indices:
                all_detections.append([
                    cls_boxes[idx, 0], cls_boxes[idx, 1],
                    cls_boxes[idx, 2], cls_boxes[idx, 3],
                    cls_scores[idx], cls
                ])

        if all_detections:
            detections = np.array(all_detections)
            order = np.argsort(-detections[:, 4])
            detections = detections[order]
            detections = detections[:max_detections]
            batch_detections.append(detections)
        else:
            batch_detections.append(np.empty((0, 6)))

    return batch_detections


def yolo_dfl_decode(
    output: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 100,
    num_classes: int = 80,
    reg_max: int = 16,
) -> List[np.ndarray]:
    """Decode YOLOv8/v10/v11 DFL (Distribution Focal Loss) output.

    YOLOv8+ output format: [batch, num_detections, 4 * reg_max + num_classes]
    where the first 4 * reg_max values are box distribution predictions.

    Args:
        output: Model output as numpy array [batch, num_detections, 4 * reg_max + num_classes].
        conf_threshold: Confidence threshold.
        iou_threshold: IoU threshold for NMS.
        max_detections: Maximum detections per image.
        num_classes: Number of classes.
        reg_max: Number of bins for DFL.

    Returns:
        List of detections per image. Each detection is [N, 6].
    """
    if output.ndim == 2:
        output = output[np.newaxis, :, :]

    batch_detections = []
    for i in range(output.shape[0]):
        pred = output[i]

        box_dist = pred[:, :4 * reg_max].reshape(-1, 4, reg_max)
        class_scores = pred[:, 4 * reg_max:]

        e_x = np.exp(box_dist - np.max(box_dist, axis=-1, keepdims=True))
        box_probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
        weights = np.arange(reg_max, dtype=np.float32)
        box_coords = np.sum(box_probs * weights, axis=-1)

        boxes = box_coords

        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        valid = max_scores >= conf_threshold
        if not np.any(valid):
            batch_detections.append(np.empty((0, 6)))
            continue

        boxes = boxes[valid]
        scores = max_scores[valid]
        class_ids = class_ids[valid]

        all_detections = []
        unique_classes = np.unique(class_ids)

        for cls in unique_classes:
            cls_mask = class_ids == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]

            indices = nms(cls_boxes, cls_scores, iou_threshold=iou_threshold)

            for idx in indices:
                all_detections.append([
                    cls_boxes[idx, 0], cls_boxes[idx, 1],
                    cls_boxes[idx, 2], cls_boxes[idx, 3],
                    cls_scores[idx], cls
                ])

        if all_detections:
            detections = np.array(all_detections)
            order = np.argsort(-detections[:, 4])
            detections = detections[order[:max_detections]]
            batch_detections.append(detections)
        else:
            batch_detections.append(np.empty((0, 6)))

    return batch_detections
