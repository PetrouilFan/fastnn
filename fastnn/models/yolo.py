"""YOLO model wrapper for fastnn.

Provides a high-level interface for loading and running YOLO models
exported to ONNX format.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import fastnn as fnn

logger = logging.getLogger(__name__)


class YOLO:
    """YOLO object detection model wrapper.

    Loads a YOLO model from ONNX, runs inference, and post-processes
    detections with NMS.

    Supports YOLOv5, YOLOv8, YOLOv10, YOLOv11 formats.

    Example:
        >>> model = YOLO("yolov8n.onnx")
        >>> results = model("image.jpg")
        >>> for det in results[0]:
        ...     x1, y1, x2, y2, conf, cls = det
    """

    def __init__(
        self,
        onnx_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        classes: Optional[List[int]] = None,
        device: str = "cpu",
    ):
        """Initialize YOLO model from ONNX file.

        Args:
            onnx_path: Path to ONNX model file.
            conf_threshold: Confidence threshold for detections.
            iou_threshold: IoU threshold for NMS.
            max_detections: Maximum detections per image.
            classes: Optional filter for class IDs.
            device: Device to run on ("cpu" or "gpu").
        """
        self.onnx_path = str(onnx_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.classes = classes
        self.device = device

        # Model state
        self.model = None
        self.input_shape = None
        self.input_name = None
        self.output_names = None
        self.num_classes = 80
        self.model_type = None  # "v5", "v8", "v10", "v11"

        # Load and prepare model
        self._load_model()

    def _load_model(self):
        """Load the ONNX model and prepare for inference."""
        # Import ONNX model to .fnn format
        import tempfile
        tmpdir = tempfile.mkdtemp()
        fnn_path = Path(tmpdir) / "model.fnn"

        try:
            # Import the ONNX model
            info = fnn.convert_from_onnx(self.onnx_path, str(fnn_path))

            # Store model info
            self.input_shape = info.get("input_shape")
            self.output_names = [o["name"] for o in info.get("graph", {}).get("outputs", [])]

            if self.input_shape:
                logger.info("Model input shape: %s", self.input_shape)

            # Detect model type from output shape
            output_shape = info.get("output_shape", [])
            if len(output_shape) >= 3:
                num_detections = output_shape[1] if len(output_shape) > 1 else output_shape[0]
                num_channels = output_shape[-1]
                logger.info("Output shape: %s, %d channels", output_shape, num_channels)

            # Build the model
            from fastnn.io.graph_builder import build_model_from_fnn
            model = build_model_from_fnn(str(fnn_path))
            self.model = model

        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise

    def preprocess(self, image: Union[str, np.ndarray, "PIL.Image.Image"]) -> Tuple[np.ndarray, Tuple]:
        """Preprocess an image for model input.

        Args:
            image: Path to image file, numpy array, or PIL Image.

        Returns:
            Tuple of (preprocessed tensor [1, C, H, W], original shape (H, W)).
        """
        if isinstance(image, str):
            from PIL import Image
            image = Image.open(image).convert("RGB")

        if hasattr(image, "mode"):
            # PIL Image
            from PIL import Image
            img = np.array(image.convert("RGB"), dtype=np.float32)
        elif isinstance(image, np.ndarray):
            img = image.astype(np.float32)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        orig_shape = img.shape[:2]  # (H, W)

        # Resize with letterbox
        if self.input_shape:
            target_h, target_w = self.input_shape[2], self.input_shape[3]
        else:
            target_h, target_w = 640, 640

        img = self._letterbox(img, target_h, target_w)

        # Normalize to [0, 1]
        img /= 255.0

        # HWC -> CHW and add batch dimension
        img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]

        return img, orig_shape

    def _letterbox(self, img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Resize image with letterbox padding to maintain aspect ratio."""
        h, w = img.shape[:2]
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize
        from PIL import Image
        pil_img = Image.fromarray(img.astype(np.uint8))
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        resized = np.array(pil_img, dtype=np.float32)

        # Pad to target size
        dh = target_h - new_h
        dw = target_w - new_w
        top, left = dh // 2, dw // 2
        bottom, right = dh - top, dw - left

        if img.ndim == 3:
            padded = np.pad(resized, ((top, bottom), (left, right), (0, 0)), mode=114/255.0)
        else:
            padded = np.pad(resized, ((top, bottom), (left, right)), mode='constant', constant_values=114/255.0)

        return padded

    def postprocess(
        self,
        outputs: Dict[str, np.ndarray],
        orig_shape: Tuple[int, int],
    ) -> List[np.ndarray]:
        """Post-process model outputs into detections.

        Args:
            outputs: Dict of output name -> numpy array.
            orig_shape: Original image shape (H, W).

        Returns:
            List of detections, each [N, 6] with [x1, y1, x2, y2, conf, cls].
        """
        # Get the main output tensor
        if self.output_names:
            output_key = self.output_names[0]
            pred = outputs.get(output_key)
        else:
            # Fallback: take first output
            pred = next(iter(outputs.values()))

        if pred is None:
            logger.warning("No output found in model results")
            return [np.empty((0, 6))]

        # Convert fastnn tensor to numpy if needed
        if hasattr(pred, "numpy"):
            pred = pred.numpy()

        # Decode based on model type
        from fastnn.utils.nms import yolo_decode, yolo_dfl_decode

        # Try DFL decode first (v8/v10/v11), fall back to standard (v5)
        try:
            num_channels = pred.shape[-1] if pred.ndim >= 2 else 0
            # v8+ output: [batch, num_dets, 4*reg_max + num_classes]
            # v5 output: [batch, num_dets, 4 + num_classes]
            reg_max = 16
            if num_channels > 4 + 80 and pred.shape[-1] >= 4 * reg_max:
                detections = yolo_dfl_decode(
                    pred,
                    conf_threshold=self.conf_threshold,
                    iou_threshold=self.iou_threshold,
                    max_detections=self.max_detections,
                )
            else:
                detections = yolo_decode(
                    pred,
                    conf_threshold=self.conf_threshold,
                    iou_threshold=self.iou_threshold,
                    max_detections=self.max_detections,
                )
        except Exception as e:
            logger.warning("YOLO decode failed: %s", e)
            detections = [np.empty((0, 6))]

        # Scale boxes to original image size
        if self.input_shape:
            scaled_detections = []
            for det in detections:
                if len(det) > 0:
                    from fastnn.utils.nms import scale_boxes
                    det[:, :4] = scale_boxes(
                        (self.input_shape[2], self.input_shape[3]),
                        det[:, :4],
                        orig_shape,
                    )
                scaled_detections.append(det)
            detections = scaled_detections

        # Filter by class if specified
        if self.classes is not None and len(detections) > 0:
            filtered = []
            for det in detections:
                if len(det) > 0:
                    mask = np.isin(det[:, 5].astype(int), self.classes)
                    filtered.append(det[mask])
                else:
                    filtered.append(det)
            detections = filtered

        return detections

    def forward(self, input_tensor: Any) -> Dict[str, Any]:
        """Run model forward pass.

        Args:
            input_tensor: Preprocessed input tensor (fastnn tensor or numpy array).

        Returns:
            Dict of output name -> tensor.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if hasattr(self.model, "forward"):
            # DAGModel
            outputs = self.model.forward({self.input_name or "images": input_tensor})
        else:
            # Sequential model
            outputs = self.model(input_tensor)
            if not isinstance(outputs, dict):
                outputs = {"output": outputs}

        return outputs

    def __call__(self, image: Union[str, np.ndarray, "PIL.Image.Image"], **kwargs) -> List[np.ndarray]:
        """Run full inference pipeline on an image.

        Args:
            image: Image path, numpy array, or PIL Image.
            **kwargs: Overrides for conf_threshold, iou_threshold, max_detections.

        Returns:
            List of detections, each [N, 6] with [x1, y1, x2, y2, conf, cls].
        """
        # Apply kwargs overrides
        conf_threshold = kwargs.get("conf_threshold", self.conf_threshold)
        iou_threshold = kwargs.get("iou_threshold", self.iou_threshold)
        max_detections = kwargs.get("max_detections", self.max_detections)

        # Store originals to restore
        orig_conf = self.conf_threshold
        orig_iou = self.iou_threshold
        orig_max = self.max_detections

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

        try:
            # Preprocess
            input_tensor, orig_shape = self.preprocess(image)

            # Convert to fastnn tensor if needed
            if not hasattr(input_tensor, "shape") or not hasattr(input_tensor, "numpy"):
                import fastnn as fnn
                input_tensor = fnn.tensor(input_tensor, list(input_tensor.shape))

            # Forward
            outputs = self.forward(input_tensor)

            # Convert outputs to numpy
            numpy_outputs = {}
            for name, tensor in outputs.items():
                if hasattr(tensor, "numpy"):
                    numpy_outputs[name] = tensor.numpy()
                else:
                    numpy_outputs[name] = np.array(tensor)

            # Postprocess
            detections = self.postprocess(numpy_outputs, orig_shape)

            return detections
        finally:
            self.conf_threshold = orig_conf
            self.iou_threshold = orig_iou
            self.max_detections = orig_max


def load_yolo(path: str, **kwargs) -> YOLO:
    """Convenience function to load a YOLO model."""
    return YOLO(path, **kwargs)
