import numpy as np


def normalize_conv_attrs(attrs: dict, keys: list[str] | None = None) -> dict:
    """Normalize ONNX convolution/pooling attribute names to internal naming.

    Args:
        attrs: dict of attribute name -> value
        keys: optional subset of keys to normalize (default: all known mappings)

    Returns:
        dict with normalized keys
    """
    mapping = {
        "kernel_shape": "kernel_size",
        "strides": "stride",
        "pads": "padding",
        "dilations": "dilation",
        "auto_pad": "auto_pad",
    }
    result = {}
    for k, v in attrs.items():
        mapped = mapping.get(k, k)
        if keys is not None and mapped not in keys:
            continue
        result[mapped] = v
    return result


def nms_filter_by_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.45,
    score_threshold: float = 0.25,
    max_detections: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class NMS filtering. Shared helper for YOLO decode functions.

    Args:
        boxes: (N, 4) array of candidate boxes
        scores: (N,) array of confidence scores
        class_ids: (N,) array of class IDs
        iou_threshold: IoU threshold for NMS
        score_threshold: minimum score to keep a detection
        max_detections: max total detections to return

    Returns:
        tuple of (boxes, scores, class_ids) after NMS
    """
    from fastnn.utils.nms import nms  # lazy import to avoid circular

    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    all_detections = []
    for cls_id in np.unique(class_ids):
        cls_mask = class_ids == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        if len(cls_boxes) == 0:
            continue

        keep = nms(cls_boxes, cls_scores, iou_threshold)
        if len(keep) == 0:
            continue

        for idx in keep:
            all_detections.append((cls_boxes[idx], cls_scores[idx], cls_id))

    if not all_detections:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    all_detections = sorted(all_detections, key=lambda x: x[1], reverse=True)[:max_detections]
    boxes_out = np.array([d[0] for d in all_detections])
    scores_out = np.array([d[1] for d in all_detections])
    class_ids_out = np.array([d[2] for d in all_detections], dtype=int)
    return boxes_out, scores_out, class_ids_out


def pack_quantized_u32(data: np.ndarray, bit_width: int) -> np.ndarray:
    """Pack quantized integer data into u32 words.

    Args:
        data: 1D array of integer values (e.g., uint8, uint4 range)
        bit_width: 4 or 8 — items per u32 word

    Returns:
        1D array of packed u32 values
    """
    items_per_word = 32 // bit_width
    mask = (1 << bit_width) - 1

    if len(data) == 0:
        return np.array([], dtype=np.uint32)

    pad = (items_per_word - len(data) % items_per_word) % items_per_word
    if pad:
        data = np.pad(data, (0, pad))

    packed = np.zeros(len(data) // items_per_word, dtype=np.uint32)
    for i in range(items_per_word):
        packed |= (data[i::items_per_word].astype(np.uint32) & mask) << (i * bit_width)

    return packed


def read_tensor_loop(
    reader,
    count: int,
    read_tensor_fn,
) -> dict:
    """Read a sequence of tensors from a binary file.

    Args:
        reader: a file-like object with read(n) method
        count: number of tensors to read
        read_tensor_fn: callable(reader) -> (name, data, dtype, scales, zeros)

    Returns:
        dict of name -> (data, dtype, scales, zeros)
    """
    params = {}
    for _ in range(count):
        name, data, dtype, scales, zeros = read_tensor_fn(reader)
        params[name] = (data, dtype, scales, zeros)
    return params


def magic_bytes() -> bytes:
    """Return the FNN file format magic bytes."""
    return b"FNN\x00"


def write_header(fp, version: int, extra: dict | None = None) -> None:
    """Write FNN file header (magic + version + optional JSON header).

    Args:
        fp: writable binary file-like object
        version: file format version (int)
        extra: optional dict to serialize as JSON header
    """
    fp.write(magic_bytes())
    import json

    if extra:
        header = {"version": version, **extra}
    else:
        header = {"version": version}
    fp.write(json.dumps(header).encode())
    fp.write(b"\n")
