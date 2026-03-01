import base64
import io
import os

import gridfs
from bson import ObjectId
from dotenv import load_dotenv
from PIL import Image
from pymongo import MongoClient
import streamlit as st

from brand_theme import inject_glass_css, render_footer, render_top_nav

BRAND = "VibeCheck"

st.set_page_config(page_title=f"{BRAND} | Delete Garments", layout="wide")
inject_glass_css(hide_sidebar=True)
render_top_nav(active="app")


@st.cache_resource
def mongo():
    load_dotenv(override=True)
    uri = os.environ.get("MONGO_URI", "").strip()
    db_name = os.environ.get("MONGO_DB", "Wardrobe_db").strip()
    if not uri:
        raise RuntimeError("Missing MONGO_URI in .env")
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    db = client[db_name]
    db.command("ping")
    fs = gridfs.GridFS(db)
    return db, fs


@st.cache_data(show_spinner=False)
def fs_get_bytes(uri: str, db_name: str, file_id_str: str) -> bytes:
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    db = client[db_name]
    fs = gridfs.GridFS(db)
    return fs.get(ObjectId(file_id_str)).read()


def get_image_from_fs(file_id_str: str) -> Image.Image:
    load_dotenv(override=True)
    uri = os.environ.get("MONGO_URI", "").strip()
    db_name = os.environ.get("MONGO_DB", "Wardrobe_db").strip()
    data = fs_get_bytes(uri, db_name, file_id_str)
    return Image.open(io.BytesIO(data))


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_image_card(img: Image.Image, caption: str = ""):
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    b64 = base64.b64encode(image_to_png_bytes(img)).decode("ascii")
    st.markdown(
        f"""
<div class="card" style="padding:8px;">
  <div class="media-frame" style="aspect-ratio:4/5;">
    <img src="data:image/png;base64,{b64}" alt="{caption}" />
  </div>
  <p class="muted" style="margin:10px 0 0 0; font-size:13px;">{caption}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def delete_garment_and_related_outfits(db, fs, customer_id: str, garment_doc: dict):
    part = garment_doc.get("part")
    gid = str(garment_doc.get("_id"))
    field = {"shirt": "shirt_id", "pants": "pants_id", "shoes": "shoes_id"}.get(part)

    deleted_outfits = 0
    if field:
        deleted_outfits = db["Outfits"].delete_many({"customer_id": customer_id, field: gid}).deleted_count

    db["Wardrobe"].delete_one({"_id": garment_doc["_id"], "customer_id": customer_id})

    img_fs_id = garment_doc.get("image_fs_id")
    if img_fs_id:
        try:
            fs.delete(ObjectId(str(img_fs_id)))
        except Exception:
            pass

    return deleted_outfits


st.markdown(
    """
<div class="landing-shell page-shell">
  <div class="glass-panel hero-panel">
    <p class="eyebrow">Account Tools</p>
    <h1 style="margin:10px 0 8px 0; font-size:1.7rem;">Delete Garments</h1>
    <p class="hero-sub">Delete garments safely. Related saved outfits will be removed automatically.</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

auth_user = st.session_state.get("auth_user")
if auth_user is None:
    st.warning("Login is required to manage garments.")
    st.page_link("wardrobe_app_auth.py", label="Go to Login")
    st.stop()

customer_id = str(auth_user["_id"])

try:
    db, fs = mongo()
except Exception as e:
    st.error(f"MongoDB connection failed: {e}")
    st.stop()

f1, f2 = st.columns([1, 1])
with f1:
    part_filter = st.selectbox("Garment type", ["all", "shirt", "pants", "shoes"], key="delete_part_filter")
with f2:
    limit = st.slider("Items shown", 10, 200, 40, 10, key="delete_limit")

q = {"customer_id": customer_id}
if part_filter != "all":
    q["part"] = part_filter

garments = list(db["Wardrobe"].find(q).sort("created_at", -1).limit(int(limit)))

if not garments:
    st.markdown('<div class="page-shell card"><p class="muted" style="margin:0;">No garments found for this filter.</p></div>', unsafe_allow_html=True)
else:
    st.caption(f"{len(garments)} garment(s)")

for g in garments:
    c1, c2, c3 = st.columns([1.2, 1.6, 1.2], gap="large")
    with c1:
        try:
            render_image_card(get_image_from_fs(g["image_fs_id"]), caption=g.get("part", "item"))
        except Exception:
            st.markdown('<div class="card"><p class="muted">Image unavailable.</p></div>', unsafe_allow_html=True)

    with c2:
        st.markdown(
            f"""
<div class="card">
  <p class="kicker">Garment</p>
  <p style="font-size:1.05rem; margin:6px 0; font-weight:700;">{g.get('part', 'item').title()}</p>
  <p class="muted" style="margin:0 0 8px 0;">Tags: {g.get('tags', [])}</p>
  <p class="muted" style="margin:0;">Created: {g.get('created_at')}</p>
</div>
""",
            unsafe_allow_html=True,
        )

    with c3:
        field = {"shirt": "shirt_id", "pants": "pants_id", "shoes": "shoes_id"}.get(g.get("part"))
        ref_count = db["Outfits"].count_documents({"customer_id": customer_id, field: str(g["_id"])}) if field else 0
        st.markdown(
            f"""
<div class="card">
  <p class="kicker">Impact</p>
  <p class="muted" style="margin:8px 0 16px 0;">Related outfits: {ref_count}</p>
</div>
""",
            unsafe_allow_html=True,
        )
        confirm_key = f"confirm_delete_{g.get('_id')}"
        st.checkbox("I understand this cannot be undone", key=confirm_key)
        if st.button("Delete Garment", key=f"delete_{g.get('_id')}", use_container_width=True):
            if not st.session_state.get(confirm_key, False):
                st.error("Please confirm deletion first.")
            else:
                deleted_outfits = delete_garment_and_related_outfits(db, fs, customer_id, g)
                st.success(f"Garment deleted. Removed {deleted_outfits} related outfit(s).")
                st.rerun()

    st.divider()

render_footer()
