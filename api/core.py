from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import os
import random
import requests
import threading
import tempfile
import types
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import bcrypt
import gridfs
import joblib
import numpy as np
from bson import ObjectId
from bson.binary import Binary
from dotenv import load_dotenv
from PIL import Image
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    models = None  # type: ignore
    transforms = None  # type: ignore

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PCA_PATH = ROOT_DIR / "work" / "model_out" / "pca_v2.joblib"
MODEL_MLP_PATH = ROOT_DIR / "work" / "model_out" / "mlp.pt"
MODEL_MLP_NUMPY_PATH = ROOT_DIR / "work" / "model_out" / "mlp_numpy.npz"
RUNTIME_TMP_ROOT = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
PENDING_DIR = RUNTIME_TMP_ROOT / "looklux_pending"
TMP_DIR = RUNTIME_TMP_ROOT / "looklux_tmp"

PARTS = {"shirt": "top", "pants": "pants", "shoes": "feet"}
PART_ORDER = ["shirt", "pants", "shoes"]
TAG_OPTIONS = ["sport", "casual", "formal", "work", "street", "summer", "winter"]
GARMENT_SIMILARITY_WARN_THRESHOLD = 0.70
DISPLAY_BG_RGB = (18, 18, 18)
TERMS_VERSION = "2026-02-28"
PRIVACY_VERSION = "2026-02-28"
LEGAL_CONSENT_TEXT = "I agree to the LookLux Terms of Use and Privacy Policy."
LEGAL_CONSENT_TEXT_HASH = hashlib.sha256(
    f"{LEGAL_CONSENT_TEXT}|terms:{TERMS_VERSION}|privacy:{PRIVACY_VERSION}".encode("utf-8")
).hexdigest()

load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)
load_dotenv(dotenv_path=ROOT_DIR / ".env.local", override=True)

INFERENCE_BASE_URL = os.environ.get("LOOKLUX_INFERENCE_URL", "").strip()
INFERENCE_TIMEOUT_SEC = int(os.environ.get("LOOKLUX_INFERENCE_TIMEOUT_SEC", "60"))


def get_config_value(key: str, default: str = "") -> str:
    value = os.environ.get(key, default)
    if isinstance(value, str):
        return value.strip()
    return default


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def ensure_dirs() -> None:
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def _import_local_human_parser() -> tuple[Any, dict[str, int]]:
    local_pkg_dir = ROOT_DIR / "fashn_human_parser"
    parser_py = local_pkg_dir / "parser.py"
    labels_py = local_pkg_dir / "labels.py"
    if not parser_py.exists() or not labels_py.exists():
        raise RuntimeError(f"Local parser files missing under {local_pkg_dir}")

    pkg_name = "fashn_human_parser_localpkg"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(local_pkg_dir)]

    labels_name = f"{pkg_name}.labels"
    labels_spec = importlib.util.spec_from_file_location(labels_name, str(labels_py))
    if labels_spec is None or labels_spec.loader is None:
        raise RuntimeError("Failed to construct labels import spec")
    labels_mod = importlib.util.module_from_spec(labels_spec)
    labels_spec.loader.exec_module(labels_mod)

    parser_name = f"{pkg_name}.parser"
    parser_spec = importlib.util.spec_from_file_location(parser_name, str(parser_py))
    if parser_spec is None or parser_spec.loader is None:
        raise RuntimeError("Failed to construct parser import spec")
    parser_mod = importlib.util.module_from_spec(parser_spec)
    parser_spec.loader.exec_module(parser_mod)

    return parser_mod.FashnHumanParser, labels_mod.LABELS_TO_IDS


try:
    FashnHumanParser, LABELS_TO_IDS = _import_local_human_parser()
    PARSER_IMPORT_ERROR = None
except Exception as local_error:
    try:
        from fashn_human_parser import FashnHumanParser, LABELS_TO_IDS  # type: ignore

        PARSER_IMPORT_ERROR = None
    except Exception as pkg_error:
        FashnHumanParser = None
        LABELS_TO_IDS = {}
        PARSER_IMPORT_ERROR = local_error or pkg_error


def l2(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def encode_vec(vec: np.ndarray) -> dict[str, Any]:
    v16 = vec.astype(np.float16)
    return {"emb_bin": Binary(v16.tobytes()), "emb_dtype": "float16", "emb_dim": int(v16.shape[0])}


def decode_vec(doc: dict[str, Any]) -> np.ndarray:
    dtype = np.float16 if doc.get("emb_dtype") == "float16" else np.float32
    return np.frombuffer(doc["emb_bin"], dtype=dtype).astype(np.float32)


def pil_rgba_to_rgb_on_white(pil_img: Image.Image) -> Image.Image:
    if pil_img.mode == "RGBA":
        bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, pil_img).convert("RGB")
    return pil_img.convert("RGB")


def pil_rgba_to_rgb_on_bg(pil_img: Image.Image, bg_rgb: tuple[int, int, int] = DISPLAY_BG_RGB) -> Image.Image:
    if pil_img.mode == "RGBA":
        bg = Image.new("RGBA", pil_img.size, (*bg_rgb, 255))
        return Image.alpha_composite(bg, pil_img).convert("RGB")
    return pil_img.convert("RGB")


def cutout_part_rgba(img_rgba: Image.Image, seg: np.ndarray, label_name: str, crop: bool = True) -> Image.Image | None:
    label_id = LABELS_TO_IDS[label_name]
    mask = (seg == label_id).astype(np.uint8) * 255
    if mask.sum() == 0:
        return None

    rgba = np.array(img_rgba.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[:, :, 3]
    rgba[:, :, 3] = np.where(mask > 0, alpha, 0).astype(np.uint8)
    masked = Image.fromarray(rgba, mode="RGBA")
    ys, xs = np.where(mask > 0)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return masked.crop((x0, y0, x1, y1)) if crop else masked


def cutout_part_bbox_rgba(img_rgba: Image.Image, seg: np.ndarray, label_name: str, crop: bool = True) -> Image.Image | None:
    label_id = LABELS_TO_IDS[label_name]
    mask = (seg == label_id).astype(np.uint8) * 255
    if mask.sum() == 0:
        return None
    ys, xs = np.where(mask > 0)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    if crop:
        return img_rgba.crop((x0, y0, x1, y1)).convert("RGBA")
    return img_rgba.convert("RGBA")


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def image_to_data_uri(img: Image.Image) -> str:
    b64 = base64.b64encode(image_to_png_bytes(img)).decode("ascii")
    return f"data:image/png;base64,{b64}"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized_png_bytes_from_bytes(img_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    return image_to_png_bytes(img)


def normalized_png_bytes_from_pil(img: Image.Image) -> bytes:
    return image_to_png_bytes(img.convert("RGBA"))


def compute_upload_sha256(img_bytes: bytes) -> str:
    try:
        return sha256_hex(normalized_png_bytes_from_bytes(img_bytes))
    except Exception:
        return sha256_hex(img_bytes)


def make_triptych(imgs: dict[str, Image.Image]) -> Image.Image:
    width = 240
    height = 240
    canvas = Image.new("RGB", (width * 3, height), DISPLAY_BG_RGB)
    for index, part in enumerate(PART_ORDER):
        item = pil_rgba_to_rgb_on_bg(imgs[part].copy(), DISPLAY_BG_RGB)
        item.thumbnail((width, height))
        tile = Image.new("RGB", (width, height), DISPLAY_BG_RGB)
        x = (width - item.size[0]) // 2
        y = (height - item.size[1]) // 2
        tile.paste(item, (x, y))
        canvas.paste(tile, (index * width, 0))
    return canvas


def fmt_score_100(score01: float) -> str:
    return f"{max(0.0, min(1.0, float(score01))) * 100.0:.1f}/100"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


@lru_cache(maxsize=1)
def load_models() -> tuple[str, Any, Any, Any, Any, dict[str, np.ndarray]]:
    if not MODEL_PCA_PATH.exists():
        raise RuntimeError(f"Missing PCA file: {MODEL_PCA_PATH}")

    device = "cpu"
    ipca = joblib.load(MODEL_PCA_PATH)

    if MODEL_MLP_NUMPY_PATH.exists():
        with np.load(MODEL_MLP_NUMPY_PATH, allow_pickle=False) as data:
            mlp = {
                "w1": data["w1"].astype(np.float32),
                "b1": data["b1"].astype(np.float32),
                "w2": data["w2"].astype(np.float32),
                "b2": data["b2"].astype(np.float32),
                "w3": data["w3"].astype(np.float32),
                "b3": data["b3"].astype(np.float32),
            }
    elif torch is not None and MODEL_MLP_PATH.exists():
        checkpoint = torch.load(MODEL_MLP_PATH, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        mlp = {
            "w1": state_dict["0.weight"].detach().cpu().numpy().astype(np.float32),
            "b1": state_dict["0.bias"].detach().cpu().numpy().astype(np.float32),
            "w2": state_dict["3.weight"].detach().cpu().numpy().astype(np.float32),
            "b2": state_dict["3.bias"].detach().cpu().numpy().astype(np.float32),
            "w3": state_dict["5.weight"].detach().cpu().numpy().astype(np.float32),
            "b3": state_dict["5.bias"].detach().cpu().numpy().astype(np.float32),
        }
    else:
        raise RuntimeError(
            "Missing lightweight MLP weights. Expected work/model_out/mlp_numpy.npz "
            "or torch-compatible work/model_out/mlp.pt"
        )

    parser = None
    resnet = None
    preprocess = None
    if torch is not None and models is not None and transforms is not None:
        parser = FashnHumanParser(device="cpu") if FashnHumanParser is not None else None

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

    return device, parser, resnet, preprocess, ipca, mlp


def emb_from_pil(pil_img: Image.Image, device: str, resnet: Any, preprocess: Any) -> np.ndarray:
    if torch is None or resnet is None or preprocess is None:
        raise RuntimeError("Local embedding extractor is unavailable in this deployment.")
    model_input = pil_rgba_to_rgb_on_white(pil_img)
    with torch.inference_mode():
        tensor = preprocess(model_input).unsqueeze(0).to(device)
        emb = resnet(tensor).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return l2(emb)


def score_from_parts(
    shirt: np.ndarray,
    pants: np.ndarray,
    shoes: np.ndarray,
    ipca: Any,
    mlp: dict[str, np.ndarray],
    device: str,
) -> float:
    fused = np.concatenate([shirt, pants, shoes]).astype(np.float32)
    fused = l2(fused)
    z = ipca.transform(fused.reshape(1, -1)).astype(np.float32)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    z1 = np.maximum(0.0, z @ mlp["w1"].T + mlp["b1"])
    z2 = np.maximum(0.0, z1 @ mlp["w2"].T + mlp["b2"])
    logits = z2 @ mlp["w3"].T + mlp["b3"]
    logits = np.clip(logits, -60.0, 60.0)
    prob = 1.0 / (1.0 + np.exp(-logits))
    return float(prob.reshape(-1)[0])


@lru_cache(maxsize=1)
def mongo_bundle() -> tuple[MongoClient, Any, gridfs.GridFS]:
    uri = get_config_value("MONGO_URI", "")
    db_name = get_config_value("MONGO_DB", "Wardrobe_db") or "Wardrobe_db"
    if not uri:
        raise RuntimeError("Missing MONGO_URI")

    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=8000,
        connectTimeoutMS=8000,
        socketTimeoutMS=8000,
    )
    db = client[db_name]
    db.command("ping")
    fs = gridfs.GridFS(db)

    try:
        for index_name, index_info in db["Customers"].index_information().items():
            keys = [key for key, _ in index_info.get("key", [])]
            if index_info.get("unique") and keys not in (["_id"], ["email"]):
                db["Customers"].drop_index(index_name)
    except Exception:
        pass

    db["Customers"].create_index([("email", 1)], unique=True)
    db["Wardrobe"].create_index([("customer_id", 1), ("part", 1), ("created_at", -1)])
    db["Outfits"].create_index([("customer_id", 1), ("created_at", -1)])
    db["ImageHashes"].create_index([("customer_id", 1), ("sha256", 1)], unique=True)

    return client, db, fs


def get_db_fs() -> tuple[Any, gridfs.GridFS]:
    _, db, fs = mongo_bundle()
    return db, fs


def fs_get_bytes(file_id_str: str) -> bytes:
    _, _, fs = mongo_bundle()
    return fs.get(ObjectId(file_id_str)).read()


def get_image_from_fs(file_id_str: str) -> Image.Image:
    data = fs_get_bytes(file_id_str)
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGBA")


def save_image_to_fs(fs: gridfs.GridFS, img: Image.Image, filename: str) -> str:
    payload = image_to_png_bytes(img.convert("RGBA"))
    file_id = fs.put(payload, filename=filename, contentType="image/png")
    return str(file_id)


def upload_already_used(db: Any, customer_id: str, sha: str) -> bool:
    return db["ImageHashes"].find_one({"customer_id": customer_id, "sha256": sha}) is not None


def remember_upload_sha(db: Any, customer_id: str, sha: str, kind: str, filename: str) -> None:
    if upload_already_used(db, customer_id, sha):
        return
    db["ImageHashes"].insert_one(
        {
            "customer_id": customer_id,
            "sha256": sha,
            "kind": kind,
            "filename": filename,
            "created_at": now_utc(),
        }
    )


def save_garment(
    db: Any,
    fs: gridfs.GridFS,
    customer_id: str,
    part: str,
    img_part: Image.Image,
    emb: np.ndarray,
    tags: list[str],
    source: str,
) -> str:
    image_sha = sha256_hex(normalized_png_bytes_from_pil(img_part))
    duplicate = db["Wardrobe"].find_one({"customer_id": customer_id, "image_sha256": image_sha})
    if duplicate is not None:
        raise ValueError("Duplicate garment image: this exact garment image already exists in your wardrobe.")

    file_id = save_image_to_fs(fs, img_part, f"{customer_id}_{part}_{int(now_utc().timestamp())}.png")
    doc = {
        "customer_id": customer_id,
        "part": part,
        "tags": list(tags),
        "image_fs_id": file_id,
        "image_sha256": image_sha,
        "source": source,
        "created_at": now_utc(),
        **encode_vec(emb),
    }
    result = db["Wardrobe"].insert_one(doc)
    return str(result.inserted_id)

def load_wardrobe(customer_id: str, part: str | None = None, tags_filter: list[str] | tuple[str, ...] = (), limit: int = 400) -> list[dict[str, Any]]:
    db, _ = get_db_fs()
    query: dict[str, Any] = {"customer_id": customer_id}
    if part:
        query["part"] = part
    if tags_filter:
        query["tags"] = {"$in": list(tags_filter)}
    projection = {
        "_id": 1,
        "part": 1,
        "tags": 1,
        "image_fs_id": 1,
        "created_at": 1,
        "emb_bin": 1,
        "emb_dtype": 1,
        "emb_dim": 1,
    }
    return list(db["Wardrobe"].find(query, projection).sort("created_at", -1).limit(int(limit)))


def get_garment_by_id(garment_id: str | ObjectId) -> dict[str, Any] | None:
    db, _ = get_db_fs()
    try:
        object_id = garment_id if isinstance(garment_id, ObjectId) else ObjectId(str(garment_id))
    except Exception:
        return None
    return db["Wardrobe"].find_one({"_id": object_id})


def get_garments_by_ids(ids: list[str]) -> dict[str, dict[str, Any]]:
    db, _ = get_db_fs()
    object_ids = []
    for item in ids:
        try:
            object_ids.append(ObjectId(item))
        except Exception:
            continue
    if not object_ids:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in db["Wardrobe"].find({"_id": {"$in": object_ids}}):
        out[str(row["_id"])] = row
    return out


def find_most_similar_garment(customer_id: str, part: str, emb: np.ndarray, limit: int = 500) -> tuple[dict[str, Any] | None, float]:
    db, _ = get_db_fs()
    items = list(db["Wardrobe"].find({"customer_id": customer_id, "part": part}).sort("created_at", -1).limit(limit))
    if not items:
        return None, -1.0

    best_doc: dict[str, Any] | None = None
    best_similarity = -1.0
    for item in items:
        try:
            similarity = cosine(emb, decode_vec(item))
        except Exception:
            continue
        if similarity > best_similarity:
            best_doc = item
            best_similarity = float(similarity)
    return best_doc, best_similarity


def infer_tag_from_existing(customer_id: str, part: str, emb: np.ndarray, topk: int = 30) -> str | None:
    db, _ = get_db_fs()
    items = list(
        db["Wardrobe"].find({"customer_id": customer_id, "part": part, "tags": {"$exists": True, "$ne": []}}).limit(800)
    )
    if not items:
        return None

    scored = []
    for item in items:
        try:
            scored.append((cosine(emb, decode_vec(item)), item))
        except Exception:
            continue
    scored.sort(key=lambda row: row[0], reverse=True)
    top = scored[: min(topk, len(scored))]

    votes: dict[str, int] = {}
    for _, item in top:
        for tag in item.get("tags", []):
            votes[tag] = votes.get(tag, 0) + 1
    if not votes:
        return None
    return sorted(votes.items(), key=lambda row: row[1], reverse=True)[0][0]


def infer_part_from_parser(parser: Any, img_path: str) -> str | None:
    if parser is None or not LABELS_TO_IDS:
        return None
    try:
        seg = parser.predict(img_path)
    except Exception:
        return None

    counts = {
        "shirt": int(np.sum(seg == LABELS_TO_IDS["top"])),
        "pants": int(np.sum(seg == LABELS_TO_IDS["pants"])),
        "shoes": int(np.sum(seg == LABELS_TO_IDS["feet"])),
    }
    best = max(counts, key=counts.get)
    if counts[best] < 2000:
        return None
    return best


def infer_part_by_similarity(customer_id: str, emb: np.ndarray) -> str | None:
    db, _ = get_db_fs()
    best_part = None
    best_similarity = -1.0
    for part in PART_ORDER:
        items = list(db["Wardrobe"].find({"customer_id": customer_id, "part": part}).limit(250))
        for item in items:
            try:
                similarity = cosine(emb, decode_vec(item))
            except Exception:
                continue
            if similarity > best_similarity:
                best_similarity = similarity
                best_part = part
    return best_part


def _decode_data_uri_image(data: str) -> Image.Image:
    if "," in data:
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data.encode("ascii"))
    with Image.open(io.BytesIO(raw)) as img:
        return img.convert("RGBA")


def _call_remote_inference(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not INFERENCE_BASE_URL:
        raise RuntimeError("Remote inference is not configured.")
    url = f"{INFERENCE_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    response = requests.post(url, json=payload, timeout=INFERENCE_TIMEOUT_SEC)
    response.raise_for_status()
    body = response.json()
    if isinstance(body, dict) and body.get("error"):
        raise RuntimeError(str(body["error"]))
    if not isinstance(body, dict):
        raise RuntimeError("Remote inference returned an invalid response.")
    return body


def extract_parts_from_upload(upload_bytes: bytes) -> tuple[dict[str, Image.Image] | None, dict[str, np.ndarray] | None, float | str]:
    device, parser, resnet, preprocess, ipca, mlp = load_models()
    has_local_extractor = parser is not None and resnet is not None and preprocess is not None and bool(LABELS_TO_IDS)

    if has_local_extractor:
        ensure_dirs()
        tmp_path = TMP_DIR / f"upload_{uuid.uuid4().hex}.png"
        tmp_path.write_bytes(upload_bytes)
        try:
            seg = parser.predict(str(tmp_path))
            img = Image.open(io.BytesIO(upload_bytes)).convert("RGBA")

            cut_imgs: dict[str, Image.Image] = {}
            embs: dict[str, np.ndarray] = {}
            missing_parts = []
            for out_part, model_label in PARTS.items():
                cut = cutout_part_rgba(img, seg, model_label, crop=True)
                if cut is None:
                    missing_parts.append(out_part)
                    continue
                cut_imgs[out_part] = cut
                embs[out_part] = emb_from_pil(cut, device, resnet, preprocess)

            if missing_parts:
                return None, None, "Missing parts: " + ", ".join(missing_parts)

            score = score_from_parts(embs["shirt"], embs["pants"], embs["shoes"], ipca, mlp, device)
            return cut_imgs, embs, score
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    if INFERENCE_BASE_URL:
        try:
            payload = {"image_b64": base64.b64encode(upload_bytes).decode("ascii")}
            body = _call_remote_inference("extract-parts", payload)
            parts = body.get("parts") or {}
            cut_imgs: dict[str, Image.Image] = {}
            embs: dict[str, np.ndarray] = {}
            for part in PART_ORDER:
                part_obj = parts.get(part) or {}
                image_b64 = part_obj.get("image_b64")
                embedding = part_obj.get("embedding")
                if not image_b64 or embedding is None:
                    return None, None, f"Missing parts: {part}"
                cut_imgs[part] = _decode_data_uri_image(str(image_b64))
                embs[part] = np.asarray(embedding, dtype=np.float32)
            score = score_from_parts(embs["shirt"], embs["pants"], embs["shoes"], ipca, mlp, device)
            return cut_imgs, embs, score
        except Exception as error:
            return None, None, f"Remote inference failed: {error}"

    error = (
        "Image extraction is unavailable in this Vercel runtime because ML dependencies are too large "
        "for Lambda storage. Set LOOKLUX_INFERENCE_URL to a remote inference service."
    )
    if PARSER_IMPORT_ERROR is not None:
        error = f"{error} ({PARSER_IMPORT_ERROR})"
    return None, None, error


def process_single_upload(upload_bytes: bytes, customer_id: str) -> tuple[str, np.ndarray, Image.Image]:
    device, parser, resnet, preprocess, _, _ = load_models()
    has_local_extractor = resnet is not None and preprocess is not None

    if has_local_extractor:
        image_rgba = Image.open(io.BytesIO(upload_bytes)).convert("RGBA")
        ensure_dirs()
        temp_path = TMP_DIR / f"single_{uuid.uuid4().hex}.png"
        temp_path.write_bytes(upload_bytes)
        try:
            part_guess = infer_part_from_parser(parser, str(temp_path))
            emb_full = emb_from_pil(image_rgba, device, resnet, preprocess)
            if part_guess is None:
                part_guess = infer_part_by_similarity(customer_id, emb_full)
            if part_guess is None:
                part_guess = "shirt"

            emb = emb_full
            save_source_img = image_rgba
            if parser is not None and LABELS_TO_IDS and part_guess in PARTS:
                try:
                    seg = parser.predict(str(temp_path))
                    cut_masked = cutout_part_rgba(image_rgba, seg, PARTS[part_guess], crop=True)
                    cut_bbox = cutout_part_bbox_rgba(image_rgba, seg, PARTS[part_guess], crop=True)
                    if cut_masked is not None:
                        emb = emb_from_pil(cut_masked, device, resnet, preprocess)
                    if cut_bbox is not None:
                        save_source_img = cut_bbox
                except Exception:
                    pass

            return part_guess, emb, save_source_img
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    if INFERENCE_BASE_URL:
        payload = {
            "customer_id": customer_id,
            "image_b64": base64.b64encode(upload_bytes).decode("ascii"),
        }
        body = _call_remote_inference("single-garment", payload)
        part_guess = str(body.get("part_guess") or "shirt")
        emb = np.asarray(body.get("embedding") or [], dtype=np.float32)
        image_b64 = body.get("image_b64")
        if emb.size == 0 or not image_b64:
            raise RuntimeError("Remote single-garment inference returned incomplete payload.")
        save_source_img = _decode_data_uri_image(str(image_b64))
        return part_guess, emb, save_source_img

    raise RuntimeError(
        "Single-garment processing is unavailable in this Vercel runtime because ML dependencies are too large "
        "for Lambda storage. Set LOOKLUX_INFERENCE_URL to a remote inference service."
    )


def make_pending_token(prefix: str) -> str:
    ensure_dirs()
    return f"{prefix}_{uuid.uuid4().hex}"


def save_pending_image(token: str, name: str, img: Image.Image) -> str:
    ensure_dirs()
    out = PENDING_DIR / f"{token}_{name}.png"
    img.convert("RGBA").save(out, format="PNG")
    return str(out)


def save_pending_embedding(token: str, name: str, emb: np.ndarray) -> str:
    ensure_dirs()
    out = PENDING_DIR / f"{token}_{name}.npy"
    np.save(out, emb.astype(np.float32), allow_pickle=False)
    return str(out)


def load_rgba_from_path(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGBA")


def cleanup_paths(paths: list[str]) -> None:
    for item in paths:
        try:
            Path(item).unlink(missing_ok=True)
        except Exception:
            pass


def cleanup_pending_outfit_payload(payload: dict[str, Any] | None) -> None:
    if not payload:
        return
    paths: list[str] = []
    paths.extend((payload.get("cut_img_paths") or {}).values())
    paths.extend((payload.get("emb_paths") or {}).values())
    cleanup_paths(paths)


def cleanup_pending_single_payload(payload: dict[str, Any] | None) -> None:
    if not payload:
        return
    paths: list[str] = []
    if payload.get("img_path"):
        paths.append(payload["img_path"])
    if payload.get("emb_path"):
        paths.append(payload["emb_path"])
    cleanup_paths(paths)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


def register_user(name: str, email: str, password: str, accepted_terms: bool) -> tuple[bool, str]:
    db, _ = get_db_fs()
    email_norm = email.strip().lower()
    if not name.strip():
        return False, "Name required"
    if "@" not in email_norm or "." not in email_norm:
        return False, "Invalid email"
    if len(password) < 6:
        return False, "Password must be at least 6 chars"
    if not accepted_terms:
        return False, "You must accept the Terms of Use and Privacy Policy to create an account."

    doc = {
        "name": name.strip(),
        "email": email_norm,
        "password_hash": hash_password(password),
        "created_at": now_utc(),
        "terms_accepted": True,
        "terms_accepted_at": now_utc(),
        "terms_version": TERMS_VERSION,
        "privacy_version": PRIVACY_VERSION,
        "consent_text_hash": LEGAL_CONSENT_TEXT_HASH,
    }

    if db["Customers"].find_one({"email": email_norm}, {"_id": 1}) is not None:
        return False, "Email already registered"

    try:
        db["Customers"].insert_one(doc)
        return True, "Registered"
    except DuplicateKeyError as error:
        details = getattr(error, "details", None) or {}
        key_pattern = details.get("keyPattern") or {}
        key_value = details.get("keyValue") or {}
        if ("email" in key_pattern) or ("email" in key_value):
            return False, "Email already registered"

        try:
            for index_name, index_info in db["Customers"].index_information().items():
                keys = [key for key, _ in index_info.get("key", [])]
                if index_info.get("unique") and keys not in (["_id"], ["email"]):
                    db["Customers"].drop_index(index_name)
            db["Customers"].create_index([("email", 1)], unique=True)
            db["Customers"].insert_one(doc)
            return True, "Registered"
        except DuplicateKeyError:
            return False, "Email already registered"
        except Exception as retry_error:
            return False, f"Register error: {retry_error}"
    except Exception as error:
        return False, f"Register error: {error}"


def login_user(email: str, password: str) -> tuple[bool, dict[str, Any] | str]:
    db, _ = get_db_fs()
    email_norm = email.strip().lower()
    user = db["Customers"].find_one({"email": email_norm})
    if not user:
        return False, "No such user"
    if not verify_password(password, user.get("password_hash", "")):
        return False, "Wrong password"
    payload = {
        "_id": str(user.get("_id")),
        "name": user.get("name", "User"),
        "email": user.get("email", ""),
    }
    return True, payload

def delete_garment_and_related_outfits(customer_id: str, garment_doc: dict[str, Any]) -> int:
    db, fs = get_db_fs()
    part_to_field = {"shirt": "shirt_id", "pants": "pants_id", "shoes": "shoes_id"}
    part = garment_doc.get("part")
    field = part_to_field.get(part)

    deleted_outfits = 0
    if field:
        deleted_outfits = db["Outfits"].delete_many({"customer_id": customer_id, field: str(garment_doc.get("_id"))}).deleted_count

    db["Wardrobe"].delete_one({"_id": garment_doc["_id"], "customer_id": customer_id})

    image_fs_id = garment_doc.get("image_fs_id")
    if image_fs_id:
        def _delete_async() -> None:
            try:
                fs.delete(ObjectId(str(image_fs_id)))
            except Exception:
                pass

        threading.Thread(target=_delete_async, daemon=True).start()

    return int(deleted_outfits)


def save_outfit(customer_id: str, score: float, shirt_doc: dict[str, Any], pants_doc: dict[str, Any], shoes_doc: dict[str, Any], tags: list[str], source: str) -> tuple[bool, str | None]:
    db, _ = get_db_fs()
    combo = {
        "customer_id": customer_id,
        "shirt_id": str(shirt_doc["_id"]),
        "pants_id": str(pants_doc["_id"]),
        "shoes_id": str(shoes_doc["_id"]),
    }
    if db["Outfits"].find_one(combo) is not None:
        return False, "Outfit already saved."

    db["Outfits"].insert_one(
        {
            **combo,
            "score": float(score),
            "tags": list(tags),
            "source": source,
            "created_at": now_utc(),
        }
    )
    return True, None


def run_match_one(
    customer_id: str,
    start_part: str,
    start_garment_id: str,
    tags_filter: list[str],
    cand_each: int,
    threshold: float,
    top_k: int,
) -> tuple[list[dict[str, Any]], str | None]:
    if start_part not in PART_ORDER:
        return [], "Invalid start type"

    shirts = load_wardrobe(customer_id, "shirt", tags_filter, limit=400)
    pants = load_wardrobe(customer_id, "pants", tags_filter, limit=400)
    shoes = load_wardrobe(customer_id, "shoes", tags_filter, limit=400)
    if not shirts or not pants or not shoes:
        return [], "Need at least 1 shirt + 1 pants + 1 shoes in wardrobe."

    chosen = get_garment_by_id(start_garment_id)
    if chosen is None or str(chosen.get("customer_id")) != customer_id or chosen.get("part") != start_part:
        return [], "Selected garment is missing."

    device, _, _, _, ipca, mlp = load_models()

    def sample_pool(pool: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
        return random.sample(pool, min(keep, len(pool)))

    pools = {
        "shirt": sample_pool(shirts, cand_each),
        "pants": sample_pool(pants, cand_each),
        "shoes": sample_pool(shoes, cand_each),
    }
    pools[start_part] = [chosen]

    scored: list[dict[str, Any]] = []
    for shirt_doc in pools["shirt"]:
        for pants_doc in pools["pants"]:
            for shoes_doc in pools["shoes"]:
                score = score_combo_fast(shirt_doc, pants_doc, shoes_doc, ipca, mlp, device)
                if score >= threshold:
                    scored.append(
                        {
                            "score": float(score),
                            "shirt_id": str(shirt_doc["_id"]),
                            "pants_id": str(pants_doc["_id"]),
                            "shoes_id": str(shoes_doc["_id"]),
                        }
                    )

    scored.sort(key=lambda row: row["score"], reverse=True)
    return scored[:top_k], None


def run_match_two(
    customer_id: str,
    part_a: str,
    garment_a_id: str,
    part_b: str,
    garment_b_id: str,
    tags_filter: list[str],
    cand_each: int,
    threshold: float,
    top_k: int,
) -> tuple[list[dict[str, Any]], str | None]:
    if part_a not in PART_ORDER or part_b not in PART_ORDER or part_a == part_b:
        return [], "Choose 2 different garment types."

    missing = list(set(PART_ORDER) - {part_a, part_b})[0]
    pool = load_wardrobe(customer_id, missing, tags_filter, limit=400)
    if not pool:
        return [], f"No {missing} found."

    doc_a = get_garment_by_id(garment_a_id)
    doc_b = get_garment_by_id(garment_b_id)
    if doc_a is None or doc_b is None:
        return [], "One selected garment is missing."
    if str(doc_a.get("customer_id")) != customer_id or str(doc_b.get("customer_id")) != customer_id:
        return [], "Invalid garment selection."
    if doc_a.get("part") != part_a or doc_b.get("part") != part_b:
        return [], "Selected garments do not match selected types."

    device, _, _, _, ipca, mlp = load_models()
    sampled = random.sample(pool, min(cand_each, len(pool)))

    scored: list[dict[str, Any]] = []
    scored_all: list[dict[str, Any]] = []
    for missing_doc in sampled:
        docs = {"shirt": None, "pants": None, "shoes": None}
        docs[part_a] = doc_a
        docs[part_b] = doc_b
        docs[missing] = missing_doc

        score = score_combo_fast(docs["shirt"], docs["pants"], docs["shoes"], ipca, mlp, device)
        row = {
            "score": float(score),
            "shirt_id": str(docs["shirt"]["_id"]),
            "pants_id": str(docs["pants"]["_id"]),
            "shoes_id": str(docs["shoes"]["_id"]),
        }
        scored_all.append(row)
        if score >= threshold:
            scored.append(row)

    scored.sort(key=lambda row: row["score"], reverse=True)
    scored = scored[:top_k]
    message = None
    if not scored and scored_all:
        scored_all.sort(key=lambda row: row["score"], reverse=True)
        scored = scored_all[:top_k]
        message = "No outfits met Min score. Showing best available matches."

    return scored, message


def run_recommendations(
    customer_id: str,
    tags_filter: list[str],
    samples: int,
    max_outfits: int,
    threshold: float,
) -> tuple[list[dict[str, Any]], str | None]:
    shirts = load_wardrobe(customer_id, "shirt", tags_filter, limit=800)
    pants = load_wardrobe(customer_id, "pants", tags_filter, limit=800)
    shoes = load_wardrobe(customer_id, "shoes", tags_filter, limit=800)
    if not shirts or not pants or not shoes:
        return [], "Need at least 1 shirt + 1 pants + 1 shoes in wardrobe."

    device, _, _, _, ipca, mlp = load_models()

    seen: set[tuple[str, str, str]] = set()
    scored: list[dict[str, Any]] = []
    for _ in range(samples):
        shirt_doc = random.choice(shirts)
        pants_doc = random.choice(pants)
        shoes_doc = random.choice(shoes)

        combo = (str(shirt_doc["_id"]), str(pants_doc["_id"]), str(shoes_doc["_id"]))
        if combo in seen:
            continue
        seen.add(combo)

        score = score_combo_fast(shirt_doc, pants_doc, shoes_doc, ipca, mlp, device)
        if score >= threshold:
            scored.append(
                {
                    "score": float(score),
                    "shirt_id": combo[0],
                    "pants_id": combo[1],
                    "shoes_id": combo[2],
                }
            )

    scored.sort(key=lambda row: row["score"], reverse=True)
    return scored[: max(1, int(max_outfits))], None


def score_combo_fast(shirt_doc: dict[str, Any], pants_doc: dict[str, Any], shoes_doc: dict[str, Any], ipca: Any, mlp: dict[str, np.ndarray], device: str) -> float:
    shirt_vec = decode_vec(shirt_doc)
    pants_vec = decode_vec(pants_doc)
    shoes_vec = decode_vec(shoes_doc)
    return score_from_parts(shirt_vec, pants_vec, shoes_vec, ipca, mlp, device)


def delete_outfit(customer_id: str, outfit_id: str) -> bool:
    db, _ = get_db_fs()
    try:
        oid = ObjectId(outfit_id)
    except Exception:
        return False
    deleted = db["Outfits"].delete_one({"_id": oid, "customer_id": customer_id}).deleted_count
    return bool(deleted)


def list_saved_outfits(customer_id: str, min_score: float, style_filter: list[str]) -> list[dict[str, Any]]:
    db, _ = get_db_fs()
    raw = list(
        db["Outfits"]
        .find(
            {"customer_id": customer_id},
            {
                "_id": 1,
                "shirt_id": 1,
                "pants_id": 1,
                "shoes_id": 1,
                "score": 1,
                "tags": 1,
                "source": 1,
                "created_at": 1,
            },
        )
        .sort("created_at", -1)
        .limit(500)
    )

    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in raw:
        combo = (item.get("shirt_id"), item.get("pants_id"), item.get("shoes_id"))
        if combo in seen:
            continue
        seen.add(combo)
        deduped.append(item)

    filtered: list[dict[str, Any]] = []
    for item in deduped:
        score = float(item.get("score", 0.0))
        if score < min_score:
            continue
        if style_filter:
            tags = item.get("tags", []) or []
            if not any(tag in tags for tag in style_filter):
                continue
        filtered.append(item)

    return filtered


def get_related_outfit_counts(customer_id: str, garment_refs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
    db, _ = get_db_fs()
    counts: dict[tuple[str, str], int] = {}

    shirt_ids = [garment_id for part, garment_id in garment_refs if part == "shirt"]
    pants_ids = [garment_id for part, garment_id in garment_refs if part == "pants"]
    shoes_ids = [garment_id for part, garment_id in garment_refs if part == "shoes"]

    if shirt_ids:
        for row in db["Outfits"].aggregate(
            [
                {"$match": {"customer_id": customer_id, "shirt_id": {"$in": shirt_ids}}},
                {"$group": {"_id": "$shirt_id", "count": {"$sum": 1}}},
            ]
        ):
            counts[("shirt", str(row["_id"]))] = int(row["count"])

    if pants_ids:
        for row in db["Outfits"].aggregate(
            [
                {"$match": {"customer_id": customer_id, "pants_id": {"$in": pants_ids}}},
                {"$group": {"_id": "$pants_id", "count": {"$sum": 1}}},
            ]
        ):
            counts[("pants", str(row["_id"]))] = int(row["count"])

    if shoes_ids:
        for row in db["Outfits"].aggregate(
            [
                {"$match": {"customer_id": customer_id, "shoes_id": {"$in": shoes_ids}}},
                {"$group": {"_id": "$shoes_id", "count": {"$sum": 1}}},
            ]
        ):
            counts[("shoes", str(row["_id"]))] = int(row["count"])

    return counts
