import io
import os
import random
import uuid
import importlib.util
import types
import colorsys
import threading
from datetime import datetime, timezone
from pathlib import Path
import base64
import hashlib
from contextlib import contextmanager
import sys

import numpy as np
from PIL import Image

import streamlit as st
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, ServerSelectionTimeoutError
import gridfs
from bson.binary import Binary
from bson import ObjectId

import bcrypt

import torch
import torch.nn as nn
from torchvision import models, transforms
import joblib

def _import_local_human_parser():
    local_pkg_dir = Path(__file__).resolve().parent / "fashn_human_parser"
    parser_py = local_pkg_dir / "parser.py"
    labels_py = local_pkg_dir / "labels.py"
    if not parser_py.exists() or not labels_py.exists():
        raise RuntimeError(f"Local parser files missing under {local_pkg_dir}")

    pkg_name = "fashn_human_parser_localpkg"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(local_pkg_dir)]
    sys.modules[pkg_name] = pkg

    labels_name = f"{pkg_name}.labels"
    labels_spec = importlib.util.spec_from_file_location(labels_name, str(labels_py))
    if labels_spec is None or labels_spec.loader is None:
        raise RuntimeError("Failed to construct labels import spec.")
    labels_mod = importlib.util.module_from_spec(labels_spec)
    sys.modules[labels_name] = labels_mod
    labels_spec.loader.exec_module(labels_mod)

    parser_name = f"{pkg_name}.parser"
    parser_spec = importlib.util.spec_from_file_location(parser_name, str(parser_py))
    if parser_spec is None or parser_spec.loader is None:
        raise RuntimeError("Failed to construct parser import spec.")
    parser_mod = importlib.util.module_from_spec(parser_spec)
    sys.modules[parser_name] = parser_mod
    parser_spec.loader.exec_module(parser_mod)

    return parser_mod.FashnHumanParser, labels_mod.LABELS_TO_IDS


try:
    # Prefer the local cv2-free parser module from this repository.
    FashnHumanParser, LABELS_TO_IDS = _import_local_human_parser()
    PARSER_IMPORT_ERROR = None
    print("[LookLux] Using local fashn_human_parser module.")
except Exception as local_e:
    try:
        from fashn_human_parser import FashnHumanParser, LABELS_TO_IDS
        PARSER_IMPORT_ERROR = None
        print("[LookLux] Using installed fashn_human_parser package.")
    except Exception as pkg_e:
        FashnHumanParser = None
        LABELS_TO_IDS = {}
        PARSER_IMPORT_ERROR = local_e if local_e is not None else pkg_e


# =========================
# CONFIG
# =========================
MODEL_PCA_PATH = "work/model_out/pca_v2.joblib"
MODEL_MLP_PATH = "work/model_out/mlp.pt"

PARTS = {"shirt": "top", "pants": "pants", "shoes": "feet"}
PART_ORDER = ["shirt", "pants", "shoes"]

DEFAULT_THRESHOLD = 0.80
DEFAULT_TOPK = 20
GARMENT_SIMILARITY_WARN_THRESHOLD = 0.70

TAG_OPTIONS = ["sport", "casual", "formal", "work", "street", "summer", "winter"]

# display background to avoid any "white" in cutouts
DISPLAY_BG_RGB = (18, 18, 18)

APP_BRAND = "LookLux"
LEGAL_CONTACT_EMAIL = "eitanmamedov7@gmail.com"
POLICY_VERSION = "2026-02-28"
TERMS_VERSION = POLICY_VERSION
PRIVACY_VERSION = POLICY_VERSION
LEGAL_CONSENT_TEXT = f"I agree to the {APP_BRAND} Terms of Use and Privacy Policy."
LEGAL_CONSENT_TEXT_HASH = hashlib.sha256(
    f"{LEGAL_CONSENT_TEXT}|terms:{TERMS_VERSION}|privacy:{PRIVACY_VERSION}".encode("utf-8")
).hexdigest()


def get_config_value(key: str, default: str = "") -> str:
    try:
        val = st.secrets.get(key, None)
    except Exception:
        val = None

    if isinstance(val, str):
        val = val.strip()
    if val:
        return val

    if load_dotenv is not None:
        try:
            load_dotenv(override=True)
        except Exception:
            pass

    env_val = os.environ.get(key, default)
    if isinstance(env_val, str):
        env_val = env_val.strip()
    return env_val

def now_utc():
    return datetime.now(timezone.utc)


def get_memory_mb():
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return float(rss) / (1024.0 * 1024.0)
        return float(rss) / 1024.0
    except Exception:
        return None


PENDING_DIR = Path("work/_pending")


def ensure_pending_dir() -> Path:
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    return PENDING_DIR


def make_pending_token(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def save_pending_image(token: str, name: str, img: Image.Image) -> str:
    out = ensure_pending_dir() / f"{token}_{name}.png"
    img.convert("RGBA").save(out, format="PNG")
    return str(out)


def save_pending_embedding(token: str, name: str, emb: np.ndarray) -> str:
    out = ensure_pending_dir() / f"{token}_{name}.npy"
    np.save(out, emb.astype(np.float32), allow_pickle=False)
    return str(out)


def load_rgba_from_path(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGBA")


def cleanup_paths(paths: list[str]):
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass


def cleanup_pending_outfit_payload(payload: dict | None):
    if not payload:
        return
    paths = []
    paths.extend((payload.get("cut_img_paths") or {}).values())
    paths.extend((payload.get("emb_paths") or {}).values())
    cleanup_paths(paths)


def cleanup_pending_single_payload(payload: dict | None):
    if not payload:
        return
    paths = []
    if payload.get("img_path"):
        paths.append(payload["img_path"])
    if payload.get("emb_path"):
        paths.append(payload["emb_path"])
    cleanup_paths(paths)

# =========================
# UI STYLE
# =========================
st.set_page_config(page_title="LookLux Platform", layout="wide")
st.markdown(
    """
<style>
    :root {
        --vc-bg-0: #040404;
        --vc-bg-1: #0b0b0b;
        --vc-bg-2: #151515;
        --vc-panel: rgba(20, 20, 20, 0.64);
        --vc-border: rgba(255, 255, 255, 0.16);
        --vc-text: #f2f2f2;
        --vc-muted: rgba(220, 220, 220, 0.78);
        --vc-shadow: 0 20px 52px rgba(0, 0, 0, 0.46);
        --vc-shadow-soft: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        color: var(--vc-text);
        font-family: "Segoe UI", system-ui, sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(1100px 520px at 8% 0%, rgba(255,255,255,.11), transparent 60%),
            radial-gradient(920px 520px at 92% 8%, rgba(165,165,165,.14), transparent 58%),
            linear-gradient(160deg, var(--vc-bg-2) 0%, var(--vc-bg-1) 52%, var(--vc-bg-0) 100%);
        animation: auroraShift 20s ease-in-out infinite alternate;
    }
    [data-testid="stHeader"], [data-testid="stToolbar"] {
        background: transparent !important;
    }
    [data-testid="stSidebar"] {
        background: rgba(10, 10, 10, 0.76) !important;
        border-right: 1px solid var(--vc-border);
        backdrop-filter: blur(16px);
    }
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    [data-testid="stSidebar"] * {
        color: var(--vc-text);
    }
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 1.5rem;
        max-width: 1280px;
    }
    a { color: #e7e7e7; text-decoration: none; }
    a:hover { color: #ffffff; }

    .glass-panel {
        background: linear-gradient(145deg, rgba(255,255,255,0.1), rgba(255,255,255,0.03));
        border: 1px solid var(--vc-border);
        border-radius: 26px;
        backdrop-filter: blur(18px);
        box-shadow: var(--vc-shadow);
    }
    .landing-shell {
        display: grid;
        gap: 16px;
        margin-bottom: 0.7rem;
        animation: revealSlide .65s ease-out;
    }
    .landing-nav {
        padding: 12px 14px;
        border-radius: 999px;
        display: flex;
        gap: 10px;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    .brand-pill {
        padding: 7px 12px;
        border-radius: 999px;
        font-size: 12px;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        border: 1px solid var(--vc-border);
        background: rgba(8, 8, 8, 0.9);
        color: #ffffff;
        font-weight: 700;
    }
    .nav-links {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        justify-content: center;
    }
    .nav-links a {
        text-decoration: none;
        color: var(--vc-muted);
        border: 1px solid rgba(255,255,255,0.14);
        background: rgba(7,7,7,0.72);
        padding: 10px 18px;
        min-width: 156px;
        white-space: nowrap;
        display: inline-flex;
        justify-content: center;
        align-items: center;
        border-radius: 999px;
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        transition: transform .2s ease, color .2s ease, border-color .2s ease;
    }
    .nav-links a:hover {
        transform: translateY(-2px);
        color: #fff;
        border-color: rgba(255,255,255,0.24);
    }

    .hero-panel {
        padding: 24px;
        animation: revealSlide .75s ease-out;
        margin-bottom: 14px;
    }
    .eyebrow {
        margin: 0;
        color: var(--vc-muted);
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 11px;
    }
    .hero-title {
        margin: 10px 0 8px 0;
        line-height: 1.05;
        font-size: clamp(1.7rem, 1.25rem + 1.9vw, 2.8rem);
        font-weight: 800;
    }
    .hero-sub {
        margin: 0;
        color: var(--vc-muted);
        font-size: 1rem;
        max-width: 72ch;
    }
    .section-grid {
        display: grid;
        gap: 12px;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }
    .section-panel {
        padding: 16px;
        transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
        animation: revealSlide .85s ease-out;
    }
    .section-panel:hover {
        transform: translateY(-3px);
        box-shadow: var(--vc-shadow-soft);
        border-color: rgba(255,255,255,0.26);
    }
    .section-panel h4 {
        margin: 0 0 8px 0;
        font-size: 1rem;
        letter-spacing: .02em;
    }
    .section-panel p {
        margin: 0;
        color: var(--vc-muted);
        font-size: 0.92rem;
        line-height: 1.45;
    }
    .legal-links {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 8px;
    }
    .legal-links a {
        text-decoration: none;
        color: var(--vc-text);
        border: 1px solid rgba(255,255,255,0.16);
        background: rgba(8,8,8,0.78);
        padding: 7px 11px;
        border-radius: 999px;
        font-size: 12px;
    }
    .legal-links a:hover {
        border-color: rgba(255,255,255,0.28);
        transform: translateY(-1px);
    }

    .card {
        background: var(--vc-panel);
        border: 1px solid var(--vc-border);
        border-radius: 22px;
        padding: 14px 14px;
        box-shadow: var(--vc-shadow-soft);
        backdrop-filter: blur(14px);
    }
    .pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(8, 8, 8, 0.85);
        border: 1px solid var(--vc-border);
        margin-right: 8px;
        font-size: 12px;
        color: var(--vc-text);
    }

    .traffic {
        width: 100%;
        height: 18px;
        border-radius: 999px;
        background: linear-gradient(90deg, #d63b3b 0%, #e3b93f 50%, #34b56f 100%);
        position: relative;
        border: 1px solid var(--vc-border);
        box-shadow: inset 0 0 0 1px rgba(0,0,0,0.2);
        margin-top: 8px;
        margin-bottom: 6px;
    }
    .marker {
        position: absolute;
        top: -4px;
        width: 10px;
        height: 26px;
        border-radius: 6px;
        background: rgba(255,255,255,0.92);
        box-shadow: 0 4px 14px rgba(0,0,0,0.45);
        transform: translateX(-50%);
    }
    .scoreNum { font-weight: 800; font-size: 14px; }
    .rec-box {
        border-radius: 22px;
        border: 2px solid var(--vc-border);
        padding: 18px 18px;
        background: rgba(18, 18, 18, 0.58);
        box-shadow: var(--vc-shadow-soft);
        backdrop-filter: blur(10px);
    }

    .imgbox {
        border-radius: 16px;
        padding: 6px;
        border: 2px solid var(--vc-border);
        background: rgba(0,0,0,0.10);
        margin-bottom: 8px;
        animation: fadeIn .25s ease-out;
    }
    .imgcap { color: var(--vc-muted); font-size: 13px; margin-top: 6px; }
    .inline-loader {
        display: flex;
        align-items: center;
        gap: 10px;
        border: 1px solid var(--vc-border);
        border-radius: 16px;
        padding: 10px 12px;
        background: rgba(11, 11, 11, 0.8);
        margin: 6px 0 10px 0;
        backdrop-filter: blur(10px);
    }
    .inline-loader .dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: rgba(225, 225, 225, 0.94);
        box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.42);
        animation: pulse 1.4s infinite;
    }
    .inline-loader .txt {
        font-size: 13px;
        color: var(--vc-text);
    }
    .inline-loader .sub {
        font-size: 12px;
        color: var(--vc-muted);
    }
    .stButton > button {
        border-radius: 999px !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        background: linear-gradient(145deg, rgba(26,26,26,0.95), rgba(8,8,8,0.92)) !important;
        color: #f6f6f6 !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        box-shadow: 0 8px 20px rgba(0,0,0,.3) !important;
        transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        border-color: rgba(255,255,255,.28) !important;
        box-shadow: 0 14px 30px rgba(0,0,0,.4) !important;
    }
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,.16) !important;
        background: rgba(12,12,12,.78) !important;
        color: #f2f2f2 !important;
        backdrop-filter: blur(8px);
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"],
    [data-testid="stTabs"] [role="tablist"] {
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 8px;
    }
    [data-testid="stTabs"] [data-baseweb="tab"],
    [data-testid="stTabs"] [role="tab"] {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        background: rgba(10, 10, 10, 0.76) !important;
        min-width: 165px;
        padding: 10px 12px !important;
        white-space: nowrap;
        font-size: 12px !important;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        font-weight: 700;
        color: #f4f4f4 !important;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        border-color: rgba(255, 255, 255, 0.36) !important;
        background: rgba(20, 20, 20, 0.94) !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-panel"],
    [data-testid="stTabs"] [role="tabpanel"] {
        background: var(--vc-panel);
        border: 1px solid var(--vc-border);
        border-radius: 22px;
        padding: 14px 14px 18px 14px;
        box-shadow: var(--vc-shadow-soft);
        backdrop-filter: blur(14px);
    }
    [data-testid="stSidebar"] [data-testid="stPageLink"] a {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        background: rgba(10, 10, 10, 0.76);
        padding: 11px 12px;
        min-width: 152px;
        white-space: nowrap;
        display: inline-flex;
        justify-content: center;
        align-items: center;
        font-size: 12px;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        font-weight: 700;
        color: #f4f4f4 !important;
    }
    [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {
        border-color: rgba(255, 255, 255, 0.34);
        background: rgba(20, 20, 20, 0.94);
    }
    [data-testid="stSidebar"] h3 {
        margin: 8px 0 8px 0;
        font-size: 0.92rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--vc-muted);
    }
    .sidebar-title {
        margin: 8px 0 6px;
        font-size: 1.5rem;
        font-weight: 800;
    }
    .sidebar-sub {
        margin: 0 0 8px 0;
        color: var(--vc-muted);
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .section-tight {
        margin-top: 14px;
    }

    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.42); }
        70% { transform: scale(1.15); box-shadow: 0 0 0 10px rgba(255, 255, 255, 0); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 255, 255, 0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes revealSlide {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes auroraShift {
        0% { background-position: 0% 0%, 100% 0%, 0% 0%; }
        100% { background-position: 8% 6%, 92% 4%, 0% 0%; }
    }
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# HELPERS
# =========================
def l2(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)

def encode_vec(vec: np.ndarray) -> dict:
    v16 = vec.astype(np.float16)
    return {"emb_bin": Binary(v16.tobytes()), "emb_dtype": "float16", "emb_dim": int(v16.shape[0])}

def decode_vec(doc: dict) -> np.ndarray:
    dt = np.float16 if doc.get("emb_dtype") == "float16" else np.float32
    return np.frombuffer(doc["emb_bin"], dtype=dt).astype(np.float32)

def pil_rgba_to_rgb_on_white(pil_img: Image.Image) -> Image.Image:
    # used for MODEL preprocessing only
    if pil_img.mode == "RGBA":
        bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
        pil_img = Image.alpha_composite(bg, pil_img).convert("RGB")
    else:
        pil_img = pil_img.convert("RGB")
    return pil_img

def pil_rgba_to_rgb_on_bg(pil_img: Image.Image, bg_rgb=DISPLAY_BG_RGB) -> Image.Image:
    # used for DISPLAY (avoid any white background)
    if pil_img.mode == "RGBA":
        bg = Image.new("RGBA", pil_img.size, (*bg_rgb, 255))
        pil_img = Image.alpha_composite(bg, pil_img).convert("RGB")
    else:
        pil_img = pil_img.convert("RGB")
    return pil_img

def cutout_part_rgba(img_rgba: Image.Image, seg: np.ndarray, label_name: str, crop=True):
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
    if crop:
        return masked.crop((x0, y0, x1, y1))
    return masked


def cutout_part_bbox_rgba(img_rgba: Image.Image, seg: np.ndarray, label_name: str, crop=True):
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


def legal_page_link(path: str, label: str):
    if hasattr(st, "page_link"):
        st.page_link(path, label=label)
    else:
        st.markdown(f"- {label}: open this page from the sidebar navigation.")


def render_nav_links_in_sidebar(include_delete_garments: bool):
    st.markdown("### Navigation")
    legal_page_link("pages/00_Home.py", "Home")
    app_entry_label = "App" if st.session_state.get("auth_user") is not None else "Get Access + Login + App"
    legal_page_link("wardrobe_app_auth.py", app_entry_label)
    if include_delete_garments:
        legal_page_link("pages/05_Delete_Garments.py", "Delete Garments")
    legal_page_link("pages/08_About.py", "About")


def render_legal_links_in_sidebar():
    st.markdown("### Legal")
    legal_page_link("pages/01_Privacy_Policy.py", "Privacy Policy")
    legal_page_link("pages/02_Terms_of_Use.py", "Terms of Use")
    legal_page_link("pages/03_Accessibility_Statement.py", "Accessibility Statement")
    legal_page_link("pages/04_Beta_Disclaimer.py", "Beta Disclaimer")

def score_to_hsl(score01: float) -> str:
    s = max(0.0, min(1.0, float(score01)))
    hue = 120.0 * s  # 0=red, 60=yellow, 120=green
    r, g, b = colorsys.hls_to_rgb(hue / 360.0, 0.52, 0.70)
    return f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"

def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def fmt_score_100(score01: float) -> str:
    return f"{max(0.0, min(1.0, float(score01))) * 100.0:.1f}/100"

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def normalized_png_bytes_from_bytes(img_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def normalized_png_bytes_from_pil(img: Image.Image) -> bytes:
    img = img.convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def compute_upload_sha256(img_bytes: bytes) -> str:
    try:
        return sha256_hex(normalized_png_bytes_from_bytes(img_bytes))
    except Exception:
        return sha256_hex(img_bytes)

def upload_already_used(db, customer_id: str, sha: str) -> bool:
    return db["ImageHashes"].find_one({"customer_id": customer_id, "sha256": sha}) is not None

def remember_upload_sha(db, customer_id: str, sha: str, kind: str, filename: str):
    if upload_already_used(db, customer_id, sha):
        return
    db["ImageHashes"].insert_one({
        "customer_id": customer_id,
        "sha256": sha,
        "kind": kind,
        "filename": filename,
        "created_at": now_utc(),
    })
    st.cache_data.clear()

def bordered_image(img: Image.Image, score: float = None, caption: str = "", max_width: int = 520):
    # IMPORTANT: keep alpha if exists (no forced RGB) so background is NEVER white
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")

    b = image_to_png_bytes(img)
    b64 = base64.b64encode(b).decode("ascii")
    border = "rgba(255,255,255,0.15)" if score is None else score_to_hsl(score)

    html = f"""
<div style="max-width:{max_width}px; margin: 0 auto;">
  <div class="imgbox" style="border-color:{border}">
    <img src="data:image/png;base64,{b64}" style="width:100%; height:auto; border-radius:14px; display:block;" />
    <div class="imgcap">{caption}</div>
  </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)

def traffic_bar(score01: float):
    score01 = max(0.0, min(1.0, float(score01)))
    pct = score01 * 100.0
    st.markdown(
        f"""
<div class="traffic"><div class="marker" style="left:{pct:.2f}%"></div></div>
<div class="scoreNum">Score: {pct:.1f}/100</div>
""",
        unsafe_allow_html=True,
    )

def make_triptych(imgs: dict) -> Image.Image:
    # display-only canvas; ensure NO white background anywhere
    w, h = 240, 240
    canvas = Image.new("RGB", (w * 3, h), DISPLAY_BG_RGB)
    for i, k in enumerate(PART_ORDER):
        im = pil_rgba_to_rgb_on_bg(imgs[k].copy(), DISPLAY_BG_RGB)
        im.thumbnail((w, h))
        tile = Image.new("RGB", (w, h), DISPLAY_BG_RGB)
        x = (w - im.size[0]) // 2
        y = (h - im.size[1]) // 2
        tile.paste(im, (x, y))
        canvas.paste(tile, (i * w, 0))
    return canvas


@contextmanager
def fancy_status(title: str):
    """
    Inline loading status that does not block or cover other UI parts.
    """
    class InlineStatus:
        def __init__(self, title_text: str):
            self.title = title_text
            self.sub = ""
            self.placeholder = st.empty()
            self.state = "running"
            self._render()

        def _render(self):
            done = self.state in ("complete", "error")
            dot_style = "background:#d9d9d9;" if self.state == "complete" else (
                "background:#8b8b8b;" if self.state == "error" else ""
            )
            pulse = "animation:none;" if done else ""
            html = f"""
<div class="inline-loader">
  <div class="dot" style="{dot_style}{pulse}"></div>
  <div>
    <div class="txt">{self.title}</div>
    <div class="sub">{self.sub}</div>
  </div>
</div>
"""
            self.placeholder.markdown(html, unsafe_allow_html=True)

        def write(self, x):
            self.sub = str(x)
            self._render()

        def update(self, **kwargs):
            label = kwargs.get("label")
            state = kwargs.get("state")
            if label:
                self.title = str(label)
            if state:
                self.state = str(state)
            self._render()

    status = InlineStatus(title)
    try:
        yield status
    finally:
        pass


# =========================
# LOAD MODELS (cached)
# =========================
@st.cache_resource
def load_models():
    if not Path(MODEL_PCA_PATH).exists():
        raise RuntimeError(f"Missing PCA file: {MODEL_PCA_PATH}")
    if not Path(MODEL_MLP_PATH).exists():
        raise RuntimeError(f"Missing MLP file: {MODEL_MLP_PATH}")

    print("[LookLux] Loading ML assets into cache (CPU)")
    device = "cpu"
    parser = FashnHumanParser(device="cpu") if FashnHumanParser is not None else None

    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    resnet = resnet.to(device).eval()
    for p in resnet.parameters():
        p.requires_grad = False

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ipca = joblib.load(MODEL_PCA_PATH)

    mlp = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device).eval()

    ckpt = torch.load(MODEL_MLP_PATH, map_location=device)
    mlp.load_state_dict(ckpt["state_dict"], strict=True)

    return device, parser, resnet, preprocess, ipca, mlp


@torch.inference_mode()
def emb_from_pil(pil_img: Image.Image, device: str, resnet: nn.Module, preprocess) -> np.ndarray:
    img = pil_rgba_to_rgb_on_white(pil_img)  # model input only
    x = preprocess(img).unsqueeze(0).to(device)
    e = resnet(x).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return l2(e)

@torch.inference_mode()
def score_from_parts(shirt: np.ndarray, pants: np.ndarray, shoes: np.ndarray, ipca, mlp, device: str) -> float:
    fused = np.concatenate([shirt, pants, shoes]).astype(np.float32)
    fused = l2(fused)
    z = ipca.transform(fused.reshape(1, -1)).astype(np.float32)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)

    xb = torch.tensor(z, dtype=torch.float32, device=device)
    prob = torch.sigmoid(mlp(xb)).detach().cpu().numpy().reshape(-1)[0].item()
    return float(prob)


# =========================
# MONGO (cached)
# =========================
@st.cache_resource
def mongo():
    uri = get_config_value("MONGO_URI", "")
    db_name = get_config_value("MONGO_DB", "Wardrobe_db") or "Wardrobe_db"
    if not uri:
        raise RuntimeError("Missing MONGO_URI in Streamlit secrets or .env")

    print("[LookLux] Connecting Mongo client (cached)")
    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=8000,
        connectTimeoutMS=8000,
        socketTimeoutMS=8000,
    )
    db = client[db_name]
    db.command("ping")

    fs = gridfs.GridFS(db)

    # indexes
    # Cleanup legacy unique indexes that can incorrectly block new registrations.
    # Keep only _id (default) and email as unique in Customers.
    try:
        for idx_name, idx_info in db["Customers"].index_information().items():
            keys = [k for k, _ in idx_info.get("key", [])]
            if idx_info.get("unique") and keys not in (["_id"], ["email"]):
                db["Customers"].drop_index(idx_name)
    except Exception:
        pass

    db["Customers"].create_index([("email", 1)], unique=True)
    db["Wardrobe"].create_index([("customer_id", 1), ("part", 1), ("created_at", -1)])
    db["Outfits"].create_index([("customer_id", 1), ("created_at", -1)])

    return client, db, fs


@st.cache_data(show_spinner=False, ttl=900, max_entries=64)
def fs_get_bytes(file_id_str: str) -> bytes:
    _client, _db, fs = mongo()
    data = fs.get(ObjectId(file_id_str)).read()
    return data


def get_image_from_fs(fs, file_id_str: str) -> Image.Image:
    data = fs_get_bytes(file_id_str)
    return Image.open(io.BytesIO(data))


def save_image_to_fs(fs, img: Image.Image, filename: str) -> str:
    buf = io.BytesIO()
    img = img.convert("RGBA")
    img.save(buf, format="PNG")
    buf.seek(0)
    file_id = fs.put(buf.read(), filename=filename, contentType="image/png")
    return str(file_id)


def save_garment(db, fs, customer_id: str, part: str, img_part: Image.Image, emb: np.ndarray, tags: list, source: str):
    img_sha = sha256_hex(normalized_png_bytes_from_pil(img_part))
    dup = db["Wardrobe"].find_one({"customer_id": customer_id, "image_sha256": img_sha})
    if dup is not None:
        raise ValueError("Duplicate garment image: this exact garment image already exists in your wardrobe.")

    file_id = save_image_to_fs(fs, img_part, f"{customer_id}_{part}_{int(now_utc().timestamp())}.png")
    vec = encode_vec(emb)
    doc = {
        "customer_id": customer_id,
        "part": part,
        "tags": tags,
        "image_fs_id": file_id,
        "image_sha256": img_sha,
        "source": source,
        "created_at": now_utc(),
        **vec,
    }
    res = db["Wardrobe"].insert_one(doc)
    st.cache_data.clear()
    return str(res.inserted_id)


@st.cache_data(show_spinner=False, ttl=180, max_entries=8)
def load_wardrobe(customer_id: str, part: str = None, tags_filter: tuple = (), limit: int = 400):
    _client, db, _fs = mongo()
    q = {"customer_id": customer_id}
    if part:
        q["part"] = part
    if tags_filter:
        q["tags"] = {"$in": list(tags_filter)}
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
    return list(db["Wardrobe"].find(q, projection).sort("created_at", -1).limit(int(limit)))


def get_garment_by_id(db, garment_id):
    try:
        if isinstance(garment_id, ObjectId):
            oid = garment_id
        else:
            oid = ObjectId(str(garment_id))
        return db["Wardrobe"].find_one({"_id": oid})
    except Exception:
        return None


def find_most_similar_garment(db, customer_id: str, part: str, emb: np.ndarray, limit: int = 500):
    items = list(db["Wardrobe"].find({"customer_id": customer_id, "part": part}).sort("created_at", -1).limit(limit))
    if not items:
        return None, -1.0

    best_doc, best_sim = None, -1.0
    for it in items:
        try:
            v = decode_vec(it)
            sim = cosine(emb, v)
            if sim > best_sim:
                best_doc, best_sim = it, sim
        except Exception:
            continue
    return best_doc, float(best_sim)


def delete_garment_and_related_outfits(db, fs, customer_id: str, garment_doc: dict):
    part = garment_doc.get("part")
    gid = str(garment_doc.get("_id"))
    part_to_field = {"shirt": "shirt_id", "pants": "pants_id", "shoes": "shoes_id"}
    field = part_to_field.get(part)

    deleted_outfits = 0
    if field:
        deleted_outfits = db["Outfits"].delete_many({"customer_id": customer_id, field: gid}).deleted_count

    db["Wardrobe"].delete_one({"_id": garment_doc["_id"], "customer_id": customer_id})

    img_fs_id = garment_doc.get("image_fs_id")
    if img_fs_id:
        def _delete_fs_file_async():
            try:
                fs.delete(ObjectId(str(img_fs_id)))
            except Exception:
                pass
        threading.Thread(target=_delete_fs_file_async, daemon=True).start()

    st.cache_data.clear()
    return deleted_outfits


# =========================
# AUTH
# =========================
def hash_password(pw: str) -> str:
    h = bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt())
    return h.decode("utf-8")

def verify_password(pw: str, pw_hash: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode("utf-8"), pw_hash.encode("utf-8"))
    except Exception:
        return False

def register_user(db, name: str, email: str, password: str, accepted_terms: bool):
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
        "created_at": datetime.now(timezone.utc),
        "terms_accepted": True,
        "terms_accepted_at": datetime.now(timezone.utc),
        "terms_version": TERMS_VERSION,
        "privacy_version": PRIVACY_VERSION,
        "consent_text_hash": LEGAL_CONSENT_TEXT_HASH,
    }
    # Fast pre-check to avoid misleading duplicate-key errors.
    if db["Customers"].find_one({"email": email_norm}, {"_id": 1}) is not None:
        return False, "Email already registered"

    try:
        db["Customers"].insert_one(doc)
        return True, "Registered"
    except DuplicateKeyError as e:
        # Only return "email already registered" when the duplicate is actually on email.
        details = getattr(e, "details", None) or {}
        key_pattern = details.get("keyPattern") or {}
        key_value = details.get("keyValue") or {}
        if ("email" in key_pattern) or ("email" in key_value):
            return False, "Email already registered"
        # Self-heal legacy wrong unique indexes (e.g. customer_id, consent_text_hash), then retry once.
        try:
            for idx_name, idx_info in db["Customers"].index_information().items():
                keys = [k for k, _ in idx_info.get("key", [])]
                if idx_info.get("unique") and keys not in (["_id"], ["email"]):
                    db["Customers"].drop_index(idx_name)
            db["Customers"].create_index([("email", 1)], unique=True)
            db["Customers"].insert_one(doc)
            return True, "Registered"
        except DuplicateKeyError:
            return False, "Email already registered"
        except Exception as retry_e:
            return False, f"Register error: {retry_e}"
        return False, f"Register error: duplicate key on {key_pattern or key_value}"
    except Exception as e:
        msg = str(e)
        return False, f"Register error: {msg}"

def login_user(db, email: str, password: str):
    email_norm = email.strip().lower()
    u = db["Customers"].find_one({"email": email_norm})
    if not u:
        return False, "No such user"
    if not verify_password(password, u.get("password_hash", "")):
        return False, "Wrong password"
    return True, u


# =========================
# STYLE INFERENCE (no extra model)
# =========================
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))

def infer_tag_from_existing(db, customer_id: str, part: str, emb: np.ndarray, topk: int = 30):
    items = list(db["Wardrobe"].find({"customer_id": customer_id, "part": part, "tags": {"$exists": True, "$ne": []}}).limit(800))
    if not items:
        return None
    scored = []
    for it in items:
        v = decode_vec(it)
        scored.append((cosine(emb, v), it))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: min(topk, len(scored))]
    votes = {}
    for _, it in top:
        for t in it.get("tags", []):
            votes[t] = votes.get(t, 0) + 1
    if not votes:
        return None
    return sorted(votes.items(), key=lambda x: x[1], reverse=True)[0][0]


def infer_part_from_parser(parser: FashnHumanParser, img_path: str):
    if parser is None:
        return None
    if not LABELS_TO_IDS:
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

def infer_part_by_similarity(db, customer_id: str, emb: np.ndarray):
    best_part, best_sim = None, -1.0
    for part in PART_ORDER:
        items = list(db["Wardrobe"].find({"customer_id": customer_id, "part": part}).limit(250))
        for it in items:
            v = decode_vec(it)
            sim = cosine(emb, v)
            if sim > best_sim:
                best_sim = sim
                best_part = part
    return best_part

# =========================
# CORE: extract parts
# =========================
def extract_parts_from_upload(tmp_path: str):
    device, parser, resnet, preprocess, ipca, mlp = load_models()
    if parser is None or not LABELS_TO_IDS:
        err = "Human parser is unavailable in this deployment (OpenCV/cv2 failed to load)."
        if PARSER_IMPORT_ERROR is not None:
            err = f"{err} {PARSER_IMPORT_ERROR}"
        return None, None, err
    seg = parser.predict(tmp_path)
    img = Image.open(tmp_path).convert("RGBA")

    cut_imgs, embs = {}, {}
    missing_parts = []
    for out_part, model_label in PARTS.items():
        cut_masked = cutout_part_rgba(img, seg, model_label, crop=True)
        cut_bbox = cutout_part_bbox_rgba(img, seg, model_label, crop=True)
        if cut_masked is None:
            missing_parts.append(out_part)
            continue
        cut_imgs[out_part] = cut_bbox if cut_bbox is not None else cut_masked
        embs[out_part] = emb_from_pil(cut_masked, device, resnet, preprocess)

    if missing_parts:
        return None, None, "Missing parts: " + ", ".join(missing_parts)

    score = score_from_parts(embs["shirt"], embs["pants"], embs["shoes"], ipca, mlp, device)
    return cut_imgs, embs, score


# =========================
# IMAGE-BASED PICKERS (consistent beautiful display)
# =========================
def pick_item_gallery(
    db,
    fs,
    customer_id: str,
    part: str,
    tags_filter: list,
    key: str,
    page_size: int = 12,
    show_selected_preview: bool = True,
    hide_grid_when_selected: bool = False,
):
    items = load_wardrobe(customer_id, part, tuple(tags_filter or []), limit=300)
    if not items:
        st.warning(f"No {part} items (after filters).")
        return None

    total = len(items)
    pages = max(1, (total + page_size - 1) // page_size)

    colA, colB = st.columns([1, 2])
    with colA:
        page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1, key=f"{key}_page")
    start = (page - 1) * page_size
    end = min(total, start + page_size)
    view = items[start:end]

    sel_id = st.session_state.get(key, "")
    if hide_grid_when_selected and sel_id:
        chosen = next((x for x in items if str(x["_id"]) == sel_id), None)
        if chosen:
            if show_selected_preview:
                try:
                    img = get_image_from_fs(fs, chosen["image_fs_id"])
                    st.markdown("#### Selected")
                    bordered_image(img, caption=f"{part} selected", max_width=360)
                except Exception:
                    pass
            return chosen

    cols = st.columns(4)
    for i, it in enumerate(view):
        n = start + i + 1  # 1..N numbering
        img = None
        try:
            img = get_image_from_fs(fs, it["image_fs_id"])  # keep RGBA if exists
        except Exception:
            img = None

        with cols[i % 4]:
            if img is not None:
                bordered_image(img, caption="", max_width=240)
            st.caption(f"{part} #{n}")
            if st.button(f"Select #{n}", key=f"{key}_btn_{it['_id']}"):
                st.session_state[key] = str(it["_id"])
                st.rerun()

    if sel_id:
        chosen = next((x for x in items if str(x["_id"]) == sel_id), None)
        if chosen:
            if show_selected_preview:
                try:
                    img = get_image_from_fs(fs, chosen["image_fs_id"])
                    st.markdown("#### Selected")
                    bordered_image(img, caption=f"{part} selected", max_width=360)
                except Exception:
                    pass
            return chosen

    return None


def score_combo_fast(s_doc, p_doc, f_doc, ipca, mlp, device: str):
    sv = decode_vec(s_doc)
    pv = decode_vec(p_doc)
    fv = decode_vec(f_doc)
    return score_from_parts(sv, pv, fv, ipca, mlp, device)


def show_outfit_card(fs, score: float, s_doc, p_doc, f_doc):
    col1, col2 = st.columns([2, 3])
    border = score_to_hsl(score)
    with col1:
        try:
            shirt_img = get_image_from_fs(fs, s_doc["image_fs_id"])
            pants_img = get_image_from_fs(fs, p_doc["image_fs_id"])
            shoes_img = get_image_from_fs(fs, f_doc["image_fs_id"])
            trip = make_triptych({"shirt": shirt_img, "pants": pants_img, "shoes": shoes_img})
            bordered_image(trip, score=score, caption="Outfit preview", max_width=460)
        except Exception as e:
            st.write("Image error:", e)

    with col2:
        st.markdown(f'<div class="rec-box" style="border-color:{border};">', unsafe_allow_html=True)
        traffic_bar(score)
        st.markdown("</div>", unsafe_allow_html=True)


def save_outfit(db, customer_id: str, score: float, s_doc, p_doc, f_doc, tags: list, source: str):
    combo_q = {
        "customer_id": customer_id,
        "shirt_id": str(s_doc["_id"]),
        "pants_id": str(p_doc["_id"]),
        "shoes_id": str(f_doc["_id"]),
    }
    if db["Outfits"].find_one(combo_q) is not None:
        return False, "Outfit already saved."

    doc = {
        **combo_q,
        "score": float(score),
        "tags": tags,
        "source": source,
        "created_at": now_utc(),
    }
    db["Outfits"].insert_one(doc)
    st.cache_data.clear()
    return True, None


# =========================
# APP START
# =========================
st.caption("Connecting to DB (cached)")
try:
    _mongo_client, db, fs = mongo()
except Exception as e:
    if st.session_state.get("auth_user") is None:
        with st.sidebar:
            render_nav_links_in_sidebar(include_delete_garments=False)
            render_legal_links_in_sidebar()

    st.error("Database connection failed. Please check MongoDB configuration.")
    st.caption(f"Error type: {type(e).__name__}")

    if isinstance(e, ServerSelectionTimeoutError):
        st.markdown(
            """
**MongoDB Atlas checklist**
- Confirm Streamlit secrets include `MONGO_URI` and `MONGO_DB`.
- In Atlas `Network Access`, allow Streamlit Cloud egress (or temporarily allow `0.0.0.0/0` for testing).
- Verify the Atlas DB user/password in `MONGO_URI` are correct and URL-encoded.
- Verify the Atlas cluster is running and reachable.
"""
        )
    else:
        st.caption(str(e))

    st.stop()

if st.session_state.get("auth_user") is None:
    with st.sidebar:
        render_nav_links_in_sidebar(include_delete_garments=False)
        render_legal_links_in_sidebar()

# Auth gate
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

if st.session_state.auth_user is None:
    st.markdown(
        """
<div class="landing-shell">
  <div class="landing-nav glass-panel">
    <div class="brand-pill">LookLux Access</div>
    <div class="nav-links">
      <a href="#auth-cta">Get Access</a>
      <a href="#auth-cta">Login</a>
      <a href="#about">About</a>
      <a href="#model">Model</a>
      <a href="#how">How it works</a>
    </div>
  </div>

  <div class="hero-panel glass-panel">
    <p class="eyebrow">Dark Glass Workspace</p>
    <h1 class="hero-title">A focused landing and navigation surface.</h1>
    <p class="hero-sub">
      Clean entry point, grayscale visual language, and smooth motion. Start with access,
      then move directly into the model-powered workspace.
    </p>
  </div>

  <div class="section-grid">
    <div class="section-panel glass-panel" id="about">
      <h4>About</h4>
      <p>
        This page is built as a premium access layer: clear onboarding and login actions,
        minimal clutter, and direct routing to the product workflow.
      </p>
    </div>
    <div class="section-panel glass-panel" id="model">
      <h4>The model we use</h4>
      <p>
        Recommendations run through parser + embedding extraction + PCA + MLP scoring,
        with deterministic ranking and cached inference for stable results.
      </p>
    </div>
    <div class="section-panel glass-panel" id="how">
      <h4>How it works</h4>
      <p>
        1) Get access or login. 2) Upload and parse visual inputs. 3) Score, match, and save
        outcomes in one continuous flow.
      </p>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div id="auth-cta"></div>', unsafe_allow_html=True)
    st.session_state.setdefault("auth_panel_mode", "login")

    cta1, cta2 = st.columns(2)
    with cta1:
        if st.button("Get Access", use_container_width=True, key="cta_get_access"):
            st.session_state.auth_panel_mode = "register"
            st.rerun()
    with cta2:
        if st.button("Login", use_container_width=True, key="cta_login"):
            st.session_state.auth_panel_mode = "login"
            st.rerun()

    st.divider()
    panel_mode = st.session_state.get("auth_panel_mode", "login")

    if panel_mode == "login":
        st.subheader("Login")
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Email", key="login_email")
            pw = st.text_input("Password", type="password", key="login_pw")
            submitted_login = st.form_submit_button("Login", use_container_width=True)

        if submitted_login:
            ok, res = login_user(db, email, pw)
            if ok:
                st.session_state.auth_user = res
                st.rerun()
            else:
                st.error(res)
    else:
        st.subheader("Get Access")
        with st.form("register_form", clear_on_submit=False):
            name = st.text_input("Name", key="reg_name")
            email2 = st.text_input("Email", key="reg_email")
            pw2 = st.text_input("Password", type="password", key="reg_pw")
            accepted_terms = st.checkbox(LEGAL_CONSENT_TEXT, key="reg_legal_accept")
            submitted_register = st.form_submit_button("Create account", use_container_width=True)

        legal_col1, legal_col2 = st.columns(2)
        with legal_col1:
            legal_page_link("pages/02_Terms_of_Use.py", "Terms of Use")
        with legal_col2:
            legal_page_link("pages/01_Privacy_Policy.py", "Privacy Policy")

        if submitted_register:
            ok, msg = register_user(db, name, email2, pw2, accepted_terms)
            if ok:
                email_norm = email2.strip().lower()
                new_user = db["Customers"].find_one({"email": email_norm})
                if new_user:
                    st.session_state.auth_user = new_user
                    st.rerun()
                st.session_state.auth_panel_mode = "login"
                st.error("Account created but automatic login failed. Please login.")
            else:
                st.error(msg)

    st.stop()

user = st.session_state.auth_user
customer_id = str(user["_id"])
customer_name = user.get("name", "User")
customer_email = user.get("email", "")

# session state for stable results (no refresh-loss after save)
st.session_state.setdefault("match1_results", [])
st.session_state.setdefault("match2_results", [])
st.session_state.setdefault("rec_results", [])
st.session_state.setdefault("last_toast", None)
st.session_state.setdefault("pending_outfit_extract", None)
st.session_state.setdefault("pending_single_upload", None)

with st.sidebar:
    st.markdown('<p class="eyebrow">Account</p><div class="sidebar-title">Workspace</div>', unsafe_allow_html=True)
    render_nav_links_in_sidebar(include_delete_garments=True)
    render_legal_links_in_sidebar()

    st.markdown("### Account")
    st.write(f"**{customer_name}**")
    st.markdown(f'<p class="sidebar-sub">{customer_email}</p>', unsafe_allow_html=True)
    if st.button("Logout", use_container_width=True):
        st.session_state.auth_user = None
        for k in list(st.session_state.keys()):
            if (
                k.startswith("pick_")
                or k.startswith("m1_")
                or k.startswith("m2_")
                or k.startswith("match")
                or k.startswith("save_anyway_")
            ):
                del st.session_state[k]
        st.rerun()

    st.markdown("### Filters")
    tags_filter = st.multiselect("Filter by tags (optional)", options=TAG_OPTIONS, default=[])

    threshold_pct = st.slider("Min score (0–100)", 0, 100, int(DEFAULT_THRESHOLD * 100), 1)
    threshold = threshold_pct / 100.0

    top_k = st.slider("Top K", 5, 50, DEFAULT_TOPK, 1)

    st.markdown("### Diagnostics")
    debug_mem = st.checkbox("Debug memory", value=False, key="debug_memory")
    if debug_mem:
        mem_mb = get_memory_mb()
        if mem_mb is not None:
            st.caption(f"Process RSS max: {mem_mb:.1f} MB")

    if st.button("Clear caches", use_container_width=True, key="clear_all_caches"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

st.markdown(
    """
<div class="hero-panel glass-panel">
  <p class="eyebrow">LookLux</p>
  <h1 class="hero-title">Style Engine</h1>
  <p class="hero-sub">Upload garments, run matches, and save the best outfit combinations.</p>
</div>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Add Outfit -> Wardrobe",
    "Match 1 Item",
    "Match 2 Items",
    "Recommend Outfits",
    "Saved Outfits",
])

# ---- TAB 1
with tab1:
    st.subheader("Upload outfit image (shirt + pants + shoes)")
    up = st.file_uploader("Upload outfit photo", type=["jpg", "jpeg", "png", "webp"], key="upload_outfit")

    auto_style = st.checkbox("Auto-detect style from your wardrobe tags (if possible)", value=True)
    tags_to_save = st.multiselect("Tags for saved garments (optional)", options=TAG_OPTIONS, default=[], key="tags_save")

    if up is not None:
        img_bytes = bytes(up.getbuffer())
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        bordered_image(img, score=None, caption=up.name, max_width=520)
        upload_sha = compute_upload_sha256(img_bytes)

        pending = st.session_state.get("pending_outfit_extract")
        if pending and pending.get("upload_sha") != upload_sha:
            cleanup_pending_outfit_payload(pending)
            st.session_state.pending_outfit_extract = None

        if st.button("Extract parts + Save to Wardrobe", use_container_width=True):
            if upload_already_used(db, customer_id, upload_sha):
                cleanup_pending_outfit_payload(st.session_state.get("pending_outfit_extract"))
                st.session_state.pending_outfit_extract = None
                st.error("Duplicate upload: this exact image was already uploaded (as outfit or garment).")
            else:
                with fancy_status("🧠 Analyzing outfit (segmentation → embeddings → scoring → saving)") as status:
                    status.write("1) Loading models (cached).")
                    _ = load_models()

                    status.write("2) Segmenting outfit into clothing regions.")
                    tmp_path = Path("work/_tmp_upload.png")
                    tmp_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path.write_bytes(img_bytes)

                    status.write("3) Cutting shirt/pants/shoes + computing embeddings.")
                    cut_imgs, embs, score_or_err = extract_parts_from_upload(str(tmp_path))
                    if cut_imgs is None:
                        cleanup_pending_outfit_payload(st.session_state.get("pending_outfit_extract"))
                        st.session_state.pending_outfit_extract = None
                        status.update(label="❌ Failed", state="error", expanded=True)
                        st.error(score_or_err)
                    else:
                        score = float(score_or_err)

                        status.write("4) Scoring the outfit compatibility (PCA + MLP).")
                        traffic_bar(score)

                        tags_final = list(tags_to_save)
                        if auto_style and not tags_final:
                            guess = infer_tag_from_existing(db, customer_id, "shirt", embs["shirt"])
                            if guess:
                                tags_final = [guess]
                                st.info(f"Auto style: {guess}")

                        trip = make_triptych(cut_imgs)
                        bordered_image(trip, score=score, caption="Extracted parts", max_width=520)
                        st.caption("Crops keep the original outfit background (no transparent masking in saved images).")

                        status.write("5) Checking for near-duplicate garments by vector similarity.")
                        similar_hits = {}
                        for part in PART_ORDER:
                            sim_doc, sim = find_most_similar_garment(db, customer_id, part, embs[part])
                            if sim_doc is not None and sim >= GARMENT_SIMILARITY_WARN_THRESHOLD:
                                similar_hits[part] = {"garment_id": str(sim_doc["_id"]), "similarity": float(sim)}

                        if similar_hits:
                            token = make_pending_token("outfit")
                            cut_img_paths = {}
                            emb_paths = {}
                            for part in PART_ORDER:
                                cut_img_paths[part] = save_pending_image(token, part, cut_imgs[part])
                                emb_paths[part] = save_pending_embedding(token, part, embs[part])

                            cleanup_pending_outfit_payload(st.session_state.get("pending_outfit_extract"))
                            st.session_state.pending_outfit_extract = {
                                "upload_sha": upload_sha,
                                "upload_name": up.name,
                                "score": score,
                                "tags_final": tags_final,
                                "cut_img_paths": cut_img_paths,
                                "emb_paths": emb_paths,
                                "similar_hits": similar_hits,
                            }
                            st.warning(
                                "Potential duplicate garments found. Review warnings below and choose which parts to save."
                            )
                            status.update(label="⚠️ Review needed", state="complete", expanded=True)
                        else:
                            status.write("6) Saving garments to your wardrobe.")
                            saved_new = 0
                            for part in PART_ORDER:
                                try:
                                    _ = save_garment(
                                        db, fs, customer_id, part, cut_imgs[part], embs[part], tags_final, source="outfit_upload"
                                    )
                                    saved_new += 1
                                except ValueError as e:
                                    st.warning(f"{part}: {e}")

                            if saved_new > 0:
                                remember_upload_sha(db, customer_id, upload_sha, kind="outfit_upload", filename=up.name)
                                st.success(f"Saved {saved_new}/3 garments to Wardrobe.")
                            else:
                                st.info("Nothing new saved (all parts were duplicates).")

                            st.session_state.pending_outfit_extract = None
                            status.update(label="✅ Done", state="complete", expanded=False)
                    del cut_imgs
                    del embs

        pending = st.session_state.get("pending_outfit_extract")
        if pending and pending.get("upload_sha") == upload_sha:
            st.markdown("#### Duplicate check review")
            st.caption("You can skip similar parts or explicitly confirm to save them anyway.")

            pending_imgs = {}
            for part in PART_ORDER:
                img_path = (pending.get("cut_img_paths") or {}).get(part)
                if img_path:
                    pending_imgs[part] = load_rgba_from_path(img_path)
            if len(pending_imgs) != len(PART_ORDER):
                st.error("Pending review artifacts expired. Please run extraction again.")
                cleanup_pending_outfit_payload(pending)
                st.session_state.pending_outfit_extract = None
                st.stop()
            trip = make_triptych(pending_imgs)
            bordered_image(
                trip,
                score=float(pending.get("score", 0.0)),
                caption="Extracted parts (review)",
                max_width=520,
            )
            st.caption("These review crops keep the original outfit background.")

            similar_hits = pending.get("similar_hits", {})
            for part in PART_ORDER:
                hit = similar_hits.get(part)
                if not hit:
                    continue
                sim_pct = hit["similarity"] * 100.0
                st.warning(f"{part}: looks very similar to an existing garment ({sim_pct:.2f}% cosine similarity).")

                existing_doc = get_garment_by_id(db, hit["garment_id"])
                if existing_doc is not None:
                    try:
                        existing_img = get_image_from_fs(fs, existing_doc["image_fs_id"])
                        bordered_image(existing_img, caption=f"Existing {part} in wardrobe", max_width=320)
                    except Exception:
                        pass

                st.checkbox(
                    f"Save {part} anyway (I confirm this is not the same garment)",
                    key=f"save_anyway_{part}",
                    value=False,
                )

            if st.button("Save reviewed garments", use_container_width=True, key="save_reviewed_outfit_parts"):
                saved_new = 0
                skipped = []
                tags_final = list(pending.get("tags_final", []))

                for part in PART_ORDER:
                    if part in similar_hits and not st.session_state.get(f"save_anyway_{part}", False):
                        skipped.append(part)
                        continue

                    try:
                        img_part = load_rgba_from_path(pending["cut_img_paths"][part])
                        emb = np.load(pending["emb_paths"][part], allow_pickle=False).astype(np.float32, copy=False)
                        _ = save_garment(db, fs, customer_id, part, img_part, emb, tags_final, source="outfit_upload")
                        saved_new += 1
                    except ValueError as e:
                        st.warning(f"{part}: {e}")

                if saved_new > 0:
                    remember_upload_sha(db, customer_id, upload_sha, kind="outfit_upload", filename=up.name)
                    st.success(f"Saved {saved_new}/3 garments to Wardrobe.")
                if skipped:
                    st.info("Skipped similar parts: " + ", ".join(skipped))
                if saved_new == 0 and not skipped:
                    st.info("Nothing new saved.")

                for part in PART_ORDER:
                    key = f"save_anyway_{part}"
                    if key in st.session_state:
                        del st.session_state[key]
                cleanup_pending_outfit_payload(pending)
                st.session_state.pending_outfit_extract = None

    st.divider()
    st.subheader("Add single garment (auto-detect type)")
    up2 = st.file_uploader("Upload garment photo", type=["jpg", "jpeg", "png", "webp"], key="upload_garment")
    auto_style2 = st.checkbox("Auto-detect style for this garment", value=True)
    tags2 = st.multiselect("Tags (optional)", options=TAG_OPTIONS, default=[], key="tags_manual")

    if up2 is not None:
        up2_bytes = bytes(up2.getbuffer())
        img2 = Image.open(io.BytesIO(up2_bytes)).convert("RGB")
        bordered_image(img2, score=None, caption=up2.name, max_width=420)
        upload_sha = compute_upload_sha256(up2_bytes)

        pending_single = st.session_state.get("pending_single_upload")
        if pending_single and pending_single.get("upload_sha") != upload_sha:
            cleanup_pending_single_payload(pending_single)
            st.session_state.pending_single_upload = None

        if st.button("Save garment", use_container_width=True):
            if upload_already_used(db, customer_id, upload_sha):
                cleanup_pending_single_payload(st.session_state.get("pending_single_upload"))
                st.session_state.pending_single_upload = None
                st.error("Duplicate upload: this exact image was already uploaded (as outfit or garment).")
            else:
                with fancy_status("🧠 Saving garment (type detection → embedding → save)") as status:
                    status.write("1) Loading models (cached).")
                    device, parser, resnet, preprocess, ipca, mlp = load_models()

                    status.write("2) Detecting type (parser → fallback similarity).")
                    img_rgba = Image.open(io.BytesIO(up2_bytes)).convert("RGBA")
                    tmp_path = Path("work/_tmp_single.png")
                    tmp_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path.write_bytes(up2_bytes)

                    part_guess = infer_part_from_parser(parser, str(tmp_path))
                    emb_full = emb_from_pil(img_rgba, device, resnet, preprocess)
                    if part_guess is None:
                        part_guess = infer_part_by_similarity(db, customer_id, emb_full)
                    if part_guess is None:
                        part_guess = "shirt"
                        st.warning("Could not infer type confidently. Defaulting to 'shirt'.")
                    st.info(f"Detected type: {part_guess}")

                    status.write("3) Computing garment embedding.")
                    emb_source_img = img_rgba
                    save_source_img = img_rgba
                    emb = emb_full
                    used_cropped_region = False
                    if parser is not None and LABELS_TO_IDS and part_guess in PARTS:
                        try:
                            seg_single = parser.predict(str(tmp_path))
                            cut_masked = cutout_part_rgba(img_rgba, seg_single, PARTS[part_guess], crop=True)
                            cut_bbox = cutout_part_bbox_rgba(img_rgba, seg_single, PARTS[part_guess], crop=True)
                            if cut_masked is not None:
                                emb_source_img = cut_masked
                                emb = emb_from_pil(emb_source_img, device, resnet, preprocess)
                                used_cropped_region = True
                            if cut_bbox is not None:
                                save_source_img = cut_bbox
                        except Exception:
                            pass
                    if used_cropped_region:
                        st.caption("Using cropped garment for embedding. Saved image keeps cropped background.")
                    else:
                        st.caption("Using full image for embedding (crop not available).")

                    tags_final = list(tags2)
                    if auto_style2 and not tags_final:
                        guess = infer_tag_from_existing(db, customer_id, part_guess, emb)
                        if guess:
                            tags_final = [guess]
                            st.info(f"Auto style: {guess}")

                    status.write("4) Checking for near-duplicate garment by vector similarity.")
                    sim_doc, sim = find_most_similar_garment(db, customer_id, part_guess, emb)
                    if sim_doc is not None and sim >= GARMENT_SIMILARITY_WARN_THRESHOLD:
                        token = make_pending_token("single")
                        img_path = save_pending_image(token, "img", save_source_img)
                        emb_path = save_pending_embedding(token, "emb", emb)
                        cleanup_pending_single_payload(st.session_state.get("pending_single_upload"))
                        st.session_state.pending_single_upload = {
                            "upload_sha": upload_sha,
                            "upload_name": up2.name,
                            "part_guess": part_guess,
                            "tags_final": tags_final,
                            "img_path": img_path,
                            "emb_path": emb_path,
                            "similar_hit": {"garment_id": str(sim_doc["_id"]), "similarity": float(sim)},
                        }
                        st.warning(
                            f"Potential duplicate detected for {part_guess} ({sim * 100.0:.2f}% cosine similarity)."
                        )
                        status.update(label="⚠️ Review needed", state="complete", expanded=True)
                    else:
                        status.write("5) Saving to wardrobe.")
                        try:
                            _ = save_garment(
                                db,
                                fs,
                                customer_id,
                                part_guess,
                                save_source_img,
                                emb,
                                tags_final,
                                source="manual_auto",
                            )
                            remember_upload_sha(db, customer_id, upload_sha, kind="single_garment_upload", filename=up2.name)
                            cleanup_pending_single_payload(st.session_state.get("pending_single_upload"))
                            st.session_state.pending_single_upload = None
                            st.success("Saved garment to Wardrobe.")
                            status.update(label="✅ Done", state="complete", expanded=False)
                        except ValueError as e:
                            st.error(str(e))
                            status.update(label="⚠️ Done with warnings", state="complete", expanded=False)

        pending_single = st.session_state.get("pending_single_upload")
        if pending_single and pending_single.get("upload_sha") == upload_sha:
            st.markdown("#### Single garment duplicate review")
            hit = pending_single["similar_hit"]
            st.warning(
                f"This garment is very similar to an existing one ({hit['similarity'] * 100.0:.2f}% cosine similarity)."
            )
            existing_doc = get_garment_by_id(db, hit["garment_id"])
            if existing_doc is not None:
                try:
                    existing_img = get_image_from_fs(fs, existing_doc["image_fs_id"])
                    bordered_image(existing_img, caption="Existing similar garment", max_width=320)
                except Exception:
                    pass

            st.checkbox(
                "Save anyway (I confirm this is not the same garment)",
                key="save_anyway_single",
                value=False,
            )
            review_col1, review_col2 = st.columns([1, 1])
            with review_col1:
                if st.button("Save reviewed garment", use_container_width=True, key="save_reviewed_single"):
                    if st.session_state.get("save_anyway_single", False):
                        try:
                            img_rgba = load_rgba_from_path(pending_single["img_path"])
                            emb = np.load(pending_single["emb_path"], allow_pickle=False).astype(np.float32, copy=False)
                            _ = save_garment(
                                db,
                                fs,
                                customer_id,
                                pending_single["part_guess"],
                                img_rgba,
                                emb,
                                list(pending_single.get("tags_final", [])),
                                source="manual_auto",
                            )
                            remember_upload_sha(
                                db,
                                customer_id,
                                upload_sha,
                                kind="single_garment_upload",
                                filename=up2.name,
                            )
                            st.success("Saved garment to Wardrobe.")
                        except ValueError as e:
                            st.error(str(e))
                    else:
                        st.info("Skipped saving because confirmation was not checked.")

                    if "save_anyway_single" in st.session_state:
                        del st.session_state["save_anyway_single"]
                    cleanup_pending_single_payload(pending_single)
                    st.session_state.pending_single_upload = None

            with review_col2:
                if st.button("Skip this garment", use_container_width=True, key="skip_reviewed_single"):
                    if "save_anyway_single" in st.session_state:
                        del st.session_state["save_anyway_single"]
                    cleanup_pending_single_payload(pending_single)
                    st.session_state.pending_single_upload = None
                    st.info("Skipped similar garment.")

# ---- TAB 2
with tab2:
    st.subheader("Pick 1 item and find matches (images only)")
    cand_each = st.slider("Candidates per missing part", 20, 200, 80, 10)

    start_part = st.selectbox("Choose start type", ["Choose type..."] + PART_ORDER, key="m1_start_part")
    if start_part not in PART_ORDER:
        st.info("Choose a type to display your wardrobe.")
    else:
        sel_key = f"pick_1_{start_part}"
        chosen = pick_item_gallery(
            db,
            fs,
            customer_id,
            start_part,
            tags_filter,
            key=sel_key,
            show_selected_preview=False,
            hide_grid_when_selected=True,
        )

        run_match = False
        if chosen:
            st.markdown("#### Selected garment")
            try:
                chosen_img = get_image_from_fs(fs, chosen["image_fs_id"])
                bordered_image(chosen_img, caption=f"{start_part} selected", max_width=360)
            except Exception:
                pass

            action_col, remove_col = st.columns([6, 1])
            with action_col:
                run_match = st.button("Get recommendations", use_container_width=True, key=f"{sel_key}_run")
            with remove_col:
                if st.button("X", use_container_width=True, key=f"{sel_key}_clear"):
                    st.session_state.pop(sel_key, None)
                    st.session_state.match1_results = []
                    st.rerun()

        if chosen and run_match:
            with fancy_status("🧠 Matching (sampling → scoring → ranking)") as status:
                status.write("1) Loading wardrobe pools.")
                shirts = load_wardrobe(customer_id, "shirt", tuple(tags_filter or []), limit=400)
                pants = load_wardrobe(customer_id, "pants", tuple(tags_filter or []), limit=400)
                shoes = load_wardrobe(customer_id, "shoes", tuple(tags_filter or []), limit=400)

                if not shirts or not pants or not shoes:
                    st.error("Need at least 1 shirt + 1 pants + 1 shoes in wardrobe.")
                    status.update(label="❌ Failed", state="error", expanded=True)
                else:
                    status.write("2) Loading scorer (cached).")
                    device, parser, resnet, preprocess, ipca, mlp = load_models()

                    def sample_pool(pool, keep):
                        return random.sample(pool, min(keep, len(pool)))

                    pools = {
                        "shirt": sample_pool(shirts, cand_each),
                        "pants": sample_pool(pants, cand_each),
                        "shoes": sample_pool(shoes, cand_each),
                    }
                    pools[start_part] = [chosen]

                    status.write("3) Scoring combinations.")
                    total = len(pools["shirt"]) * len(pools["pants"]) * len(pools["shoes"])
                    prog = st.progress(0)
                    done = 0

                    scored = []
                    for s_doc in pools["shirt"]:
                        for p_doc in pools["pants"]:
                            for f_doc in pools["shoes"]:
                                sc = score_combo_fast(s_doc, p_doc, f_doc, ipca, mlp, device)
                                if sc >= threshold:
                                    scored.append((sc, str(s_doc["_id"]), str(p_doc["_id"]), str(f_doc["_id"])))
                                done += 1
                                if done % 200 == 0:
                                    prog.progress(min(1.0, done / max(1, total)))

                    prog.progress(1.0)
                    scored.sort(key=lambda x: x[0], reverse=True)
                    scored = scored[:top_k]
                    st.session_state.match1_results = scored
                    st.session_state.last_toast = None

                    status.write("4) Showing top results.")
                    status.update(label="✅ Done", state="complete", expanded=False)

        if st.session_state.match1_results:
            st.success(f"Found {len(st.session_state.match1_results)} matches (top {len(st.session_state.match1_results)})")
            if st.session_state.last_toast:
                st.info(st.session_state.last_toast)

            for i, (sc, sid, pid, fid) in enumerate(st.session_state.match1_results, start=1):
                s_doc = get_garment_by_id(db, sid)
                p_doc = get_garment_by_id(db, pid)
                f_doc = get_garment_by_id(db, fid)
                if not (s_doc and p_doc and f_doc):
                    st.warning("Skipping a result because one of the garments is missing.")
                    continue

                show_outfit_card(fs, sc, s_doc, p_doc, f_doc)

                if st.button(
                    f"Save outfit (score {fmt_score_100(sc)})",
                    key=f"save1_{i}_{sid}_{pid}_{fid}",
                ):
                    ok, msg = save_outfit(db, customer_id, sc, s_doc, p_doc, f_doc, tags_filter, source="match1")
                    if ok:
                        st.session_state.last_toast = "✅ Outfit saved."
                        st.toast("Saved")
                    else:
                        st.session_state.last_toast = msg
                        st.info(msg)

# ---- TAB 3
with tab3:
    st.subheader("Pick 2 items and complete the outfit (images only)")
    cand_each2 = st.slider("Candidates for missing part", 20, 300, 120, 10)

    part_a = st.selectbox("First type", ["Choose type..."] + PART_ORDER, key="m2_part_a")
    if part_a in PART_ORDER:
        opts_b = ["Choose type..."] + [p for p in PART_ORDER if p != part_a]
    else:
        opts_b = ["Choose type..."] + PART_ORDER
    part_b = st.selectbox("Second type", opts_b, key="m2_part_b")

    if part_a not in PART_ORDER or part_b not in PART_ORDER:
        st.info("Choose 2 types to begin. No images are shown before selection.")
    else:
        cur = (part_a, part_b)
        prev = st.session_state.get("m2_prev_parts")
        if prev != cur:
            st.session_state["m2_prev_parts"] = cur
            st.session_state.pop("m2_sel_a_id", None)
            st.session_state.pop("m2_sel_b_id", None)
            st.session_state.pop("m2_confirmed", None)
            st.session_state.match2_results = []
            st.session_state.pop("m2_pick_a", None)
            st.session_state.pop("m2_pick_b", None)

        if st.button("Reset selection", use_container_width=True):
            st.session_state.pop("m2_sel_a_id", None)
            st.session_state.pop("m2_sel_b_id", None)
            st.session_state.pop("m2_confirmed", None)
            st.session_state.match2_results = []
            st.session_state.pop("m2_pick_a", None)
            st.session_state.pop("m2_pick_b", None)
            st.rerun()

        if st.session_state.get("m2_sel_a_id") is None:
            st.markdown(f"#### Step 1: pick {part_a}")
            a_doc = pick_item_gallery(db, fs, customer_id, part_a, tags_filter, key="m2_pick_a")
            if a_doc:
                st.session_state["m2_sel_a_id"] = str(a_doc["_id"])
                st.rerun()

        elif st.session_state.get("m2_sel_b_id") is None:
            st.markdown(f"#### Step 1 selected ({part_a})")
            a_doc = get_garment_by_id(db, st.session_state["m2_sel_a_id"])
            if a_doc:
                try:
                    a_img = get_image_from_fs(fs, a_doc["image_fs_id"])
                    bordered_image(a_img, caption=f"{part_a} selected", max_width=360)
                except Exception:
                    pass

            st.markdown(f"#### Step 2: pick {part_b}")
            b_doc = pick_item_gallery(db, fs, customer_id, part_b, tags_filter, key="m2_pick_b")
            if b_doc:
                st.session_state["m2_sel_b_id"] = str(b_doc["_id"])
                st.rerun()

        else:
            a_doc = get_garment_by_id(db, st.session_state["m2_sel_a_id"])
            b_doc = get_garment_by_id(db, st.session_state["m2_sel_b_id"])
            if not a_doc or not b_doc:
                st.error("Selection missing. Press Reset and choose again.")
            else:
                # SHOW BOTH SELECTED IMAGES + delete options
                st.markdown("#### Selected items")
                c1, c2 = st.columns(2)
                with c1:
                    try:
                        imgA = get_image_from_fs(fs, a_doc["image_fs_id"])
                        bordered_image(imgA, caption=f"{part_a} (selected)", max_width=360)
                    except Exception:
                        st.warning("Could not load first image.")
                    if st.button(f"Remove {part_a}", key="m2_remove_a", use_container_width=True):
                        st.session_state.pop("m2_sel_a_id", None)
                        st.session_state.pop("m2_confirmed", None)
                        st.session_state.match2_results = []
                        st.rerun()

                with c2:
                    try:
                        imgB = get_image_from_fs(fs, b_doc["image_fs_id"])
                        bordered_image(imgB, caption=f"{part_b} (selected)", max_width=360)
                    except Exception:
                        st.warning("Could not load second image.")
                    if st.button(f"Remove {part_b}", key="m2_remove_b", use_container_width=True):
                        st.session_state.pop("m2_sel_b_id", None)
                        st.session_state.pop("m2_confirmed", None)
                        st.session_state.match2_results = []
                        st.rerun()

                st.divider()

                if not st.session_state.get("m2_confirmed", False):
                    st.info("To continue, confirm the selection of both garments.")
                    if st.button("Confirm selection", use_container_width=True):
                        st.session_state["m2_confirmed"] = True
                        st.session_state.match2_results = []
                        st.rerun()
                else:
                    missing = list(set(PART_ORDER) - {part_a, part_b})[0]
                    st.success(f"Confirmed: {part_a} + {part_b}. Missing: {missing}")

                    if st.button(f"Find best {missing}", use_container_width=True):
                        with fancy_status("🧠 Completing outfit (sampling → scoring → ranking)") as status:
                            status.write("1) Loading candidates for the missing part.")
                            pool = load_wardrobe(customer_id, missing, tuple(tags_filter or []), limit=400)
                            if not pool:
                                st.error(f"No {missing} found.")
                                status.update(label="❌ Failed", state="error", expanded=True)
                            else:
                                status.write("2) Loading scorer (cached).")
                                device, parser, resnet, preprocess, ipca, mlp = load_models()

                                pool = random.sample(pool, min(cand_each2, len(pool)))
                                status.write("3) Scoring candidates.")
                                prog = st.progress(0)
                                scored = []
                                scored_all = []
                                for j, m_doc in enumerate(pool, start=1):
                                    docs = {"shirt": None, "pants": None, "shoes": None}
                                    docs[part_a] = a_doc
                                    docs[part_b] = b_doc
                                    docs[missing] = m_doc

                                    sc = score_combo_fast(docs["shirt"], docs["pants"], docs["shoes"], ipca, mlp, device)
                                    combo_row = (sc, str(docs["shirt"]["_id"]), str(docs["pants"]["_id"]), str(docs["shoes"]["_id"]))
                                    scored_all.append(combo_row)
                                    if sc >= threshold:
                                        scored.append(combo_row)

                                    if j % 20 == 0:
                                        prog.progress(min(1.0, j / max(1, len(pool))))

                                prog.progress(1.0)
                                scored.sort(key=lambda x: x[0], reverse=True)
                                scored = scored[:top_k]
                                if not scored and scored_all:
                                    scored_all.sort(key=lambda x: x[0], reverse=True)
                                    scored = scored_all[:top_k]
                                    st.session_state.last_toast = "No outfits met Min score. Showing best available matches."
                                else:
                                    st.session_state.last_toast = None
                                st.session_state.match2_results = scored

                                status.update(label="✅ Done", state="complete", expanded=False)

                    if st.session_state.match2_results:
                        st.success(f"Found {len(st.session_state.match2_results)} matches")
                        if st.session_state.last_toast:
                            st.info(st.session_state.last_toast)

                        for i, (sc, sid, pid, fid) in enumerate(st.session_state.match2_results, start=1):
                            s_doc = get_garment_by_id(db, sid)
                            p_doc = get_garment_by_id(db, pid)
                            f_doc = get_garment_by_id(db, fid)
                            if not (s_doc and p_doc and f_doc):
                                st.warning("Skipping a result because one of the garments is missing.")
                                continue

                            show_outfit_card(fs, sc, s_doc, p_doc, f_doc)
                            if st.button(
                                f"Save outfit (score {fmt_score_100(sc)})",
                                key=f"save2_{i}_{sid}_{pid}_{fid}",
                            ):
                                ok, msg = save_outfit(db, customer_id, sc, s_doc, p_doc, f_doc, tags_filter, source="match2")
                                if ok:
                                    st.session_state.last_toast = "✅ Outfit saved."
                                    st.toast("Saved")
                                else:
                                    st.session_state.last_toast = msg
                                    st.info(msg)

# ---- TAB 4
with tab4:
    st.subheader("Recommend outfits (random sampling)")
    samples = st.slider("How many samples", 200, 100000, 5000, 250)
    max_outfits = st.number_input("Max outfits to return (1..∞)", min_value=1, value=20, step=1)

    if st.button("Generate", use_container_width=True):
        with fancy_status("🧠 Recommending outfits (sampling → scoring → ranking)") as status:
            status.write("1) Loading wardrobe pools.")
            shirts = load_wardrobe(customer_id, "shirt", tuple(tags_filter or []), limit=800)
            pants = load_wardrobe(customer_id, "pants", tuple(tags_filter or []), limit=800)
            shoes = load_wardrobe(customer_id, "shoes", tuple(tags_filter or []), limit=800)

            if not shirts or not pants or not shoes:
                st.error("Need at least 1 shirt + 1 pants + 1 shoes in wardrobe.")
                status.update(label="❌ Failed", state="error", expanded=True)
            else:
                status.write("2) Loading scorer (cached).")
                device, parser, resnet, preprocess, ipca, mlp = load_models()

                status.write("3) Sampling unique combinations + scoring.")
                seen = set()
                scored = []
                prog = st.progress(0)
                kept = 0

                for t in range(samples):
                    s_doc = random.choice(shirts)
                    p_doc = random.choice(pants)
                    f_doc = random.choice(shoes)

                    combo = (str(s_doc["_id"]), str(p_doc["_id"]), str(f_doc["_id"]))
                    if combo in seen:
                        continue
                    seen.add(combo)

                    sc = score_combo_fast(s_doc, p_doc, f_doc, ipca, mlp, device)
                    if sc >= threshold:
                        scored.append((sc, combo[0], combo[1], combo[2]))
                        kept += 1

                    if t % 500 == 0 and t > 0:
                        prog.progress(min(1.0, t / max(1, samples)))

                prog.progress(1.0)
                scored.sort(key=lambda x: x[0], reverse=True)
                scored = scored[: int(max_outfits)]

                st.session_state.rec_results = scored
                st.session_state.last_toast = None

                status.write(f"4) Done. Kept {kept} above threshold, showing {len(scored)}.")
                status.update(label="✅ Done", state="complete", expanded=False)

    if st.session_state.rec_results:
        st.success(f"Found {len(st.session_state.rec_results)} outfits (unique)")
        if st.session_state.last_toast:
            st.info(st.session_state.last_toast)

        for i, (sc, sid, pid, fid) in enumerate(st.session_state.rec_results, start=1):
            s_doc = get_garment_by_id(db, sid)
            p_doc = get_garment_by_id(db, pid)
            f_doc = get_garment_by_id(db, fid)
            if not (s_doc and p_doc and f_doc):
                st.warning("Skipping a result because one of the garments is missing.")
                continue

            show_outfit_card(fs, sc, s_doc, p_doc, f_doc)
            if st.button(
                f"Save outfit (score {fmt_score_100(sc)})",
                key=f"saveR_{i}_{sid}_{pid}_{fid}",
            ):
                ok, msg = save_outfit(db, customer_id, sc, s_doc, p_doc, f_doc, tags_filter, source="recommend")
                if ok:
                    st.session_state.last_toast = "✅ Outfit saved."
                    st.toast("Saved")
                else:
                    st.session_state.last_toast = msg
                    st.info(msg)

# ---- TAB 5
with tab5:
    st.subheader("Saved outfits")
    delete_outfit_toast = st.session_state.pop("delete_outfit_toast", None)
    if delete_outfit_toast:
        st.toast(delete_outfit_toast, icon="✅")

    # filters (score + styles)
    c1, c2 = st.columns([1, 2])
    with c1:
        min_saved_score_pct = st.slider("Min saved score (0–100)", 0, 100, 0, 1, key="saved_min_score")
    with c2:
        saved_style_filter = st.multiselect("Filter saved by styles (optional)", TAG_OPTIONS, default=[], key="saved_style_filter")

    saved_raw = list(
        db["Outfits"].find(
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
        ).sort("created_at", -1).limit(500)
    )

    # de-dup by combo
    seen = set()
    saved = []
    for o in saved_raw:
        combo = (o.get("shirt_id"), o.get("pants_id"), o.get("shoes_id"))
        if combo in seen:
            continue
        seen.add(combo)
        saved.append(o)

    # filter by score + styles
    filtered = []
    for o in saved:
        sc = float(o.get("score", 0.0))
        if sc < (min_saved_score_pct / 100.0):
            continue
        if saved_style_filter:
            tags = o.get("tags", []) or []
            if not any(t in tags for t in saved_style_filter):
                continue
        filtered.append(o)

    if not filtered:
        st.info("No saved outfits matching your filters.")
    else:
        total_filtered = len(filtered)
        st.write(f"Saved (filtered): {total_filtered}")

        page_col1, page_col2 = st.columns([1, 1])
        with page_col1:
            saved_page_size = st.selectbox(
                "Saved outfits per page",
                options=[6, 12, 24, 36],
                index=0,
                key="saved_outfits_page_size",
            )
        total_pages = max(1, (total_filtered + int(saved_page_size) - 1) // int(saved_page_size))
        with page_col2:
            saved_page = int(
                st.number_input(
                    "Saved outfits page",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    step=1,
                    key="saved_outfits_page_num",
                )
            )

        start_idx = (saved_page - 1) * int(saved_page_size)
        end_idx = min(total_filtered, start_idx + int(saved_page_size))
        page_rows = filtered[start_idx:end_idx]
        st.caption(f"Showing {start_idx + 1}-{end_idx} of {total_filtered}")

        garment_ids = set()
        for o in page_rows:
            for gid in (o.get("shirt_id"), o.get("pants_id"), o.get("shoes_id")):
                if gid:
                    garment_ids.add(str(gid))

        wardrobe_docs = {}
        if garment_ids:
            oid_list = []
            for gid in garment_ids:
                try:
                    oid_list.append(ObjectId(gid))
                except Exception:
                    continue
            if oid_list:
                for d in db["Wardrobe"].find({"_id": {"$in": oid_list}}, {"_id": 1, "image_fs_id": 1}):
                    wardrobe_docs[str(d["_id"])] = d

        for o in page_rows:
            # delete button (per outfit doc)
            del_col, _ = st.columns([1, 6])
            with del_col:
                if st.button("🗑️ Delete", key=f"del_{o.get('_id')}"):
                    deleted = db["Outfits"].delete_one({"_id": o["_id"], "customer_id": customer_id}).deleted_count
                    if deleted:
                        st.session_state["delete_outfit_toast"] = "Deleted successfully."
                    else:
                        st.session_state["delete_outfit_toast"] = "Item was already deleted."
                    st.rerun()

            s_doc = wardrobe_docs.get(str(o.get("shirt_id")))
            p_doc = wardrobe_docs.get(str(o.get("pants_id")))
            f_doc = wardrobe_docs.get(str(o.get("shoes_id")))
            if not (s_doc and p_doc and f_doc):
                st.warning("This saved outfit references a missing garment. Skipping display.")
                continue

            sc = float(o.get("score", 0.0))
            show_outfit_card(fs, sc, s_doc, p_doc, f_doc)
            st.caption(f"tags: {o.get('tags', [])} | source: {o.get('source')} | saved_at: {o.get('created_at')}")

    st.divider()
    st.subheader("Garment Management")
    st.caption("Garment deletion moved to a separate page for cleaner workflow.")
    legal_page_link("pages/05_Delete_Garments.py", "Open Delete Garments page")
    
