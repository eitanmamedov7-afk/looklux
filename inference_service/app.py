from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path
from functools import lru_cache
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, jsonify, request
from torchvision import models, transforms

# Ensure local package imports work when the service runs from inference_service/.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fashn_human_parser import FashnHumanParser, LABELS_TO_IDS

PARTS = {"shirt": "top", "pants": "pants", "shoes": "feet"}
PART_ORDER = ["shirt", "pants", "shoes"]
PART_MIN_PIXELS = 2000

app = Flask(__name__)


def _decode_image_b64(payload_b64: str) -> Image.Image:
    data = payload_b64.strip()
    if "," in data:
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data.encode("ascii"))
    with Image.open(io.BytesIO(raw)) as img:
        return img.convert("RGBA")


def _image_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGBA").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _pil_rgba_to_rgb_on_white(pil_img: Image.Image) -> Image.Image:
    if pil_img.mode == "RGBA":
        bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, pil_img).convert("RGB")
    return pil_img.convert("RGB")


def _cutout_masked_rgba(img_rgba: Image.Image, seg: np.ndarray, label_name: str) -> Image.Image | None:
    label_id = LABELS_TO_IDS[label_name]
    mask = (seg == label_id).astype(np.uint8) * 255
    if int(mask.sum()) == 0:
        return None

    rgba = np.array(img_rgba.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[:, :, 3]
    rgba[:, :, 3] = np.where(mask > 0, alpha, 0).astype(np.uint8)
    masked = Image.fromarray(rgba, mode="RGBA")
    ys, xs = np.where(mask > 0)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return masked.crop((x0, y0, x1, y1))


def _cutout_bbox_rgba(img_rgba: Image.Image, seg: np.ndarray, label_name: str) -> Image.Image | None:
    label_id = LABELS_TO_IDS[label_name]
    mask = (seg == label_id).astype(np.uint8) * 255
    if int(mask.sum()) == 0:
        return None
    ys, xs = np.where(mask > 0)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return img_rgba.crop((x0, y0, x1, y1)).convert("RGBA")


@lru_cache(maxsize=1)
def _load_models() -> tuple[str, Any, Any, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = FashnHumanParser(device=device)

    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    resnet = resnet.to(device).eval()
    for parameter in resnet.parameters():
        parameter.requires_grad = False

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return device, parser, resnet, preprocess


def _l2(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def _embedding_from_pil(pil_img: Image.Image, device: str, resnet: Any, preprocess: Any) -> np.ndarray:
    rgb = _pil_rgba_to_rgb_on_white(pil_img)
    with torch.inference_mode():
        tensor = preprocess(rgb).unsqueeze(0).to(device)
        emb = resnet(tensor).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return _l2(emb)


def _part_counts(seg: np.ndarray) -> dict[str, int]:
    return {
        "shirt": int(np.sum(seg == LABELS_TO_IDS["top"])),
        "pants": int(np.sum(seg == LABELS_TO_IDS["pants"])),
        "shoes": int(np.sum(seg == LABELS_TO_IDS["feet"])),
    }


@app.get("/health")
def health() -> Any:
    return jsonify({"ok": True})


@app.get("/")
def root() -> Any:
    return jsonify(
        {
            "ok": True,
            "service": "looklux-inference",
            "endpoints": ["/health", "/extract-parts", "/single-garment"],
        }
    )


@app.post("/extract-parts")
def extract_parts() -> Any:
    try:
        payload = request.get_json(force=True, silent=False) or {}
        image_b64 = str(payload.get("image_b64") or "")
        if not image_b64:
            return jsonify({"error": "Missing image_b64"}), 400

        image_rgba = _decode_image_b64(image_b64)
        device, parser, resnet, preprocess = _load_models()
        seg = parser.predict(image_rgba)

        parts: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        for part in PART_ORDER:
            cut_masked = _cutout_masked_rgba(image_rgba, seg, PARTS[part])
            cut_bbox = _cutout_bbox_rgba(image_rgba, seg, PARTS[part])
            if cut_masked is None:
                missing.append(part)
                continue
            emb = _embedding_from_pil(cut_masked, device, resnet, preprocess)
            view_img = cut_bbox if cut_bbox is not None else cut_masked
            parts[part] = {
                "image_b64": _image_to_data_uri(view_img),
                "embedding": emb.astype(np.float32).tolist(),
            }

        if missing:
            return jsonify({"error": "Missing parts: " + ", ".join(missing)}), 400
        return jsonify({"parts": parts})
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.post("/single-garment")
def single_garment() -> Any:
    try:
        payload = request.get_json(force=True, silent=False) or {}
        image_b64 = str(payload.get("image_b64") or "")
        if not image_b64:
            return jsonify({"error": "Missing image_b64"}), 400

        image_rgba = _decode_image_b64(image_b64)
        device, parser, resnet, preprocess = _load_models()

        seg = parser.predict(image_rgba)
        counts = _part_counts(seg)
        part_guess = max(counts, key=counts.get)
        if counts[part_guess] < PART_MIN_PIXELS:
            part_guess = "shirt"

        cut_masked = _cutout_masked_rgba(image_rgba, seg, PARTS[part_guess])
        cut_bbox = _cutout_bbox_rgba(image_rgba, seg, PARTS[part_guess])

        emb_source = cut_masked if cut_masked is not None else image_rgba
        emb = _embedding_from_pil(emb_source, device, resnet, preprocess)

        save_img = cut_bbox if cut_bbox is not None else image_rgba
        return jsonify(
            {
                "part_guess": part_guess,
                "embedding": emb.astype(np.float32).tolist(),
                "image_b64": _image_to_data_uri(save_img),
            }
        )
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.post("/api/extract-parts")
def extract_parts_api_alias() -> Any:
    return extract_parts()


@app.post("/api/single-garment")
def single_garment_api_alias() -> Any:
    return single_garment()


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    app.run(host=host, port=port)
