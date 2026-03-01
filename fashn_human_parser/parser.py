"""FASHN Human Parser - SegFormer-based human parsing model (cv2-free)."""

from typing import List, Union

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import SegformerForSemanticSegmentation

from .labels import IDS_TO_LABELS

# ImageNet normalization constants (same as training)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Model input size (matches training)
INPUT_HEIGHT = 576
INPUT_WIDTH = 384


class FashnHumanParser:
    """Human parsing model that segments images into semantic classes."""

    def __init__(
        self,
        model_id: str = "fashn-ai/fashn-human-parser",
        device: str = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()

    def _preprocess_single(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a single RGB uint8 image for model input."""
        pil = Image.fromarray(image, mode="RGB")
        # BOX resampling is closest PIL equivalent to area-style downsampling.
        pil = pil.resize((INPUT_WIDTH, INPUT_HEIGHT), resample=Image.Resampling.BOX)
        resized = np.array(pil, dtype=np.uint8)

        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
        transposed = normalized.transpose(2, 0, 1)
        return torch.from_numpy(transposed)

    def _to_numpy(self, image: Union[Image.Image, np.ndarray, str]) -> np.ndarray:
        """Convert input image to RGB numpy uint8 array."""
        if isinstance(image, str):
            with Image.open(image) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                image = np.array(pil_img.convert("RGB"), dtype=np.uint8)
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"), dtype=np.uint8)

        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Unsupported image type: {type(image).__name__}. "
                "Expected PIL Image, numpy array, or file path string."
            )

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3:
            if image.shape[2] == 4:
                image = np.array(Image.fromarray(image, mode="RGBA").convert("RGB"), dtype=np.uint8)
            elif image.shape[2] != 3:
                raise ValueError(
                    f"Expected RGB image with 3 channels, got {image.shape[2]} channels"
                )
        else:
            raise ValueError(f"Expected 2D or 3D image array, got {image.ndim}D array")

        if image.dtype in (np.float32, np.float64):
            image = (image * 255).clip(0, 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)

        return image

    @torch.inference_mode()
    def predict(
        self,
        image: Union[Image.Image, np.ndarray, str, List],
        return_logits: bool = False,
    ):
        """Run human parsing on one image or a batch list."""
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]

        if len(images) == 0:
            return []
        for i, img in enumerate(images):
            if img is None:
                raise ValueError(f"Image at index {i} is None")

        images_np = [self._to_numpy(img) for img in images]
        original_sizes = [(img.shape[0], img.shape[1]) for img in images_np]  # (H, W)
        batch_tensors = [self._preprocess_single(img) for img in images_np]

        pixel_values = torch.stack(batch_tensors).to(self.device)
        model_dtype = next(self.model.parameters()).dtype
        pixel_values = pixel_values.to(dtype=model_dtype)

        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits

        results = []
        for i, size in enumerate(original_sizes):
            img_logits = logits[i : i + 1]
            upsampled = torch.nn.functional.interpolate(
                img_logits,
                size=size,
                mode="bilinear",
                align_corners=False,
            )

            if return_logits:
                results.append(upsampled)
            else:
                pred_seg = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()
                results.append(pred_seg)

        return results if is_batch else results[0]

    @staticmethod
    def get_label_name(label_id: int) -> str:
        return IDS_TO_LABELS.get(label_id, "unknown")

    @staticmethod
    def get_labels() -> dict:
        return IDS_TO_LABELS.copy()

