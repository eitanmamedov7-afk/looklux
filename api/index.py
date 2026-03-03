from __future__ import annotations

import io
import math
from functools import wraps
from typing import Any

import numpy as np
from flask import Flask, abort, flash, redirect, render_template, request, send_file, session, url_for

try:
    from . import core
except Exception:  # pragma: no cover
    import core  # type: ignore

BRAND = "LookLux"
CONTACT_EMAIL = "eitanmamedov7@gmail.com"
POLICY_DATE = "2026-02-28"

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = core.get_config_value("APP_SECRET_KEY", core.get_config_value("SESSION_SECRET", "dev-secret-change-me"))
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = core.get_config_value("COOKIE_SECURE", "0") == "1"


@app.context_processor
def inject_globals() -> dict[str, Any]:
    return {
        "brand": BRAND,
        "contact_email": CONTACT_EMAIL,
        "policy_date": POLICY_DATE,
        "tag_options": core.TAG_OPTIONS,
        "part_order": core.PART_ORDER,
    }


def get_auth_user() -> dict[str, Any] | None:
    auth = session.get("auth_user")
    if isinstance(auth, dict) and auth.get("_id"):
        return auth
    return None


def require_auth(handler):
    @wraps(handler)
    def wrapped(*args, **kwargs):
        if get_auth_user() is None:
            flash("Login is required.", "error")
            return redirect(url_for("app_page"))
        return handler(*args, **kwargs)

    return wrapped


def parse_int(value: Any, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    try:
        num = int(value)
    except Exception:
        num = default
    if min_value is not None:
        num = max(min_value, num)
    if max_value is not None:
        num = min(max_value, num)
    return num


def parse_float(value: Any, default: float, min_value: float | None = None, max_value: float | None = None) -> float:
    try:
        num = float(value)
    except Exception:
        num = default
    if min_value is not None:
        num = max(min_value, num)
    if max_value is not None:
        num = min(max_value, num)
    return num


def get_filter_state() -> dict[str, Any]:
    tags_filter = session.get("tags_filter")
    if not isinstance(tags_filter, list):
        tags_filter = []
    tags_filter = [tag for tag in tags_filter if tag in core.TAG_OPTIONS]

    threshold = parse_float(session.get("threshold"), 0.80, 0.0, 1.0)
    top_k = parse_int(session.get("top_k"), 20, 1, 100)

    return {
        "tags_filter": tags_filter,
        "threshold": threshold,
        "threshold_pct": int(round(threshold * 100)),
        "top_k": top_k,
    }


def set_filter_state_from_request() -> None:
    tags_filter = request.form.getlist("tags_filter")
    tags_filter = [tag for tag in tags_filter if tag in core.TAG_OPTIONS]
    threshold_pct = parse_int(request.form.get("threshold_pct"), 80, 0, 100)
    top_k = parse_int(request.form.get("top_k"), 20, 1, 100)

    session["tags_filter"] = tags_filter
    session["threshold"] = float(threshold_pct) / 100.0
    session["top_k"] = top_k


def format_results_for_display(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def normalize_id(value: Any) -> str:
        text = str(value or "").strip()
        if not text or text.lower() == "none":
            return ""
        return text

    ids: list[str] = []
    for row in results:
        for key in ("shirt_id", "pants_id", "shoes_id"):
            garment_id = normalize_id(row.get(key))
            if garment_id:
                ids.append(garment_id)

    docs_by_id = core.get_garments_by_ids(ids)

    cards: list[dict[str, Any]] = []
    for row in results:
        shirt_id = normalize_id(row.get("shirt_id"))
        pants_id = normalize_id(row.get("pants_id"))
        shoes_id = normalize_id(row.get("shoes_id"))
        if not (shirt_id and pants_id and shoes_id):
            continue

        shirt_doc = docs_by_id.get(shirt_id)
        pants_doc = docs_by_id.get(pants_id)
        shoes_doc = docs_by_id.get(shoes_id)

        missing_parts: list[str] = []
        if shirt_doc is None:
            missing_parts.append("shirt")
        if pants_doc is None:
            missing_parts.append("pants")
        if shoes_doc is None:
            missing_parts.append("shoes")

        cards.append(
            {
                "score": float(row.get("score", 0.0)),
                "score_label": core.fmt_score_100(float(row.get("score", 0.0))),
                "shirt_id": shirt_id,
                "pants_id": pants_id,
                "shoes_id": shoes_id,
                "shirt_image": str(shirt_doc.get("image_fs_id")) if shirt_doc and shirt_doc.get("image_fs_id") else None,
                "pants_image": str(pants_doc.get("image_fs_id")) if pants_doc and pants_doc.get("image_fs_id") else None,
                "shoes_image": str(shoes_doc.get("image_fs_id")) if shoes_doc and shoes_doc.get("image_fs_id") else None,
                "can_save": len(missing_parts) == 0,
                "missing_parts": missing_parts,
                "source": row.get("source"),
            }
        )

    return cards


def pending_outfit_preview(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    cut_paths = payload.get("cut_img_paths") or {}
    try:
        imgs = {
            "shirt": core.load_rgba_from_path(cut_paths["shirt"]),
            "pants": core.load_rgba_from_path(cut_paths["pants"]),
            "shoes": core.load_rgba_from_path(cut_paths["shoes"]),
        }
    except Exception:
        return None
    trip = core.make_triptych(imgs)
    return core.image_to_data_uri(trip)


def pending_single_preview(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    img_path = payload.get("img_path")
    if not img_path:
        return None
    try:
        image = core.load_rgba_from_path(img_path)
    except Exception:
        return None
    return core.image_to_data_uri(image)


@app.get("/")
def home_page():
    return render_template("home.html", user=get_auth_user(), active_nav="home")


@app.get("/about")
def about_page():
    return render_template("about.html", user=get_auth_user(), active_nav="about")


@app.get("/legal/privacy")
def privacy_page():
    return render_template("legal_privacy.html", user=get_auth_user(), active_nav="legal")


@app.get("/legal/terms")
def terms_page():
    return render_template("legal_terms.html", user=get_auth_user(), active_nav="legal")


@app.get("/legal/accessibility")
def accessibility_page():
    return render_template("legal_accessibility.html", user=get_auth_user(), active_nav="legal")


@app.get("/legal/beta-disclaimer")
def beta_page():
    return render_template("legal_beta.html", user=get_auth_user(), active_nav="legal")


@app.get("/media/<file_id>")
@require_auth
def media_file(file_id: str):
    try:
        data = core.fs_get_bytes(file_id)
    except Exception:
        abort(404)
    return send_file(io.BytesIO(data), mimetype="image/png")


def handle_auth_post(action: str) -> bool:
    if action == "login":
        email = request.form.get("email", "")
        password = request.form.get("password", "")
        ok, result = core.login_user(email, password)
        if ok:
            session["auth_user"] = result
            flash("Logged in.", "success")
        else:
            flash(str(result), "error")
        return True

    if action == "register":
        name = request.form.get("name", "")
        email = request.form.get("email", "")
        password = request.form.get("password", "")
        accepted_terms = request.form.get("accepted_terms") == "on"
        ok, message = core.register_user(name, email, password, accepted_terms)
        if not ok:
            flash(message, "error")
            return True

        ok_login, result = core.login_user(email, password)
        if ok_login:
            session["auth_user"] = result
            flash("Account created and logged in.", "success")
        else:
            flash("Account created. Please login.", "info")
        return True

    return False


def action_upload_outfit(user: dict[str, Any], filters: dict[str, Any]) -> None:
    file = request.files.get("upload_outfit")
    if file is None or file.filename is None or file.filename == "":
        flash("Please upload an outfit image.", "error")
        return

    img_bytes = file.read()
    if not img_bytes:
        flash("Uploaded outfit is empty.", "error")
        return

    db, fs = core.get_db_fs()
    customer_id = str(user["_id"])
    upload_sha = core.compute_upload_sha256(img_bytes)

    existing_payload = session.get("pending_outfit_extract")
    if existing_payload and existing_payload.get("upload_sha") != upload_sha:
        core.cleanup_pending_outfit_payload(existing_payload)
        session.pop("pending_outfit_extract", None)

    if core.upload_already_used(db, customer_id, upload_sha):
        core.cleanup_pending_outfit_payload(session.get("pending_outfit_extract"))
        session.pop("pending_outfit_extract", None)
        flash("Duplicate upload: this exact image was already uploaded.", "error")
        return

    cut_imgs, embs, score_or_err = core.extract_parts_from_upload(img_bytes)
    if cut_imgs is None or embs is None:
        flash(str(score_or_err), "error")
        return

    tags = request.form.getlist("tags_save")
    tags = [tag for tag in tags if tag in core.TAG_OPTIONS]
    auto_style = request.form.get("auto_style") == "on"
    if auto_style and not tags:
        guessed = core.infer_tag_from_existing(customer_id, "shirt", embs["shirt"])
        if guessed:
            tags = [guessed]
            flash(f"Auto style: {guessed}", "info")

    similar_hits: dict[str, Any] = {}
    for part in core.PART_ORDER:
        similar_doc, similarity = core.find_most_similar_garment(customer_id, part, embs[part])
        if similar_doc is not None and similarity >= core.GARMENT_SIMILARITY_WARN_THRESHOLD:
            similar_hits[part] = {"garment_id": str(similar_doc["_id"]), "similarity": float(similarity)}

    if similar_hits:
        token = core.make_pending_token("outfit")
        cut_img_paths: dict[str, str] = {}
        emb_paths: dict[str, str] = {}
        for part in core.PART_ORDER:
            cut_img_paths[part] = core.save_pending_image(token, part, cut_imgs[part])
            emb_paths[part] = core.save_pending_embedding(token, part, embs[part])

        core.cleanup_pending_outfit_payload(session.get("pending_outfit_extract"))
        session["pending_outfit_extract"] = {
            "upload_sha": upload_sha,
            "upload_name": file.filename,
            "score": float(score_or_err),
            "tags_final": tags,
            "cut_img_paths": cut_img_paths,
            "emb_paths": emb_paths,
            "similar_hits": similar_hits,
        }
        flash("Potential duplicate garments found. Review and confirm parts to save.", "warning")
        return

    saved_new = 0
    for part in core.PART_ORDER:
        try:
            core.save_garment(db, fs, customer_id, part, cut_imgs[part], embs[part], tags, source="outfit_upload")
            saved_new += 1
        except ValueError as error:
            flash(f"{part}: {error}", "warning")

    if saved_new > 0:
        core.remember_upload_sha(db, customer_id, upload_sha, kind="outfit_upload", filename=file.filename)
        flash(f"Saved {saved_new}/3 garments to wardrobe.", "success")
    else:
        flash("Nothing new saved (all parts were duplicates).", "info")

    core.cleanup_pending_outfit_payload(session.get("pending_outfit_extract"))
    session.pop("pending_outfit_extract", None)


def action_confirm_outfit_review(user: dict[str, Any]) -> None:
    payload = session.get("pending_outfit_extract")
    if not payload:
        flash("No pending outfit review found.", "error")
        return

    db, fs = core.get_db_fs()
    customer_id = str(user["_id"])
    similar_hits = payload.get("similar_hits") or {}
    saved_new = 0
    skipped: list[str] = []

    for part in core.PART_ORDER:
        if part in similar_hits and request.form.get(f"save_anyway_{part}") != "on":
            skipped.append(part)
            continue

        try:
            image = core.load_rgba_from_path(payload["cut_img_paths"][part])
            emb = np.load(payload["emb_paths"][part], allow_pickle=False).astype(np.float32, copy=False)
            core.save_garment(
                db,
                fs,
                customer_id,
                part,
                image,
                emb,
                list(payload.get("tags_final", [])),
                source="outfit_upload",
            )
            saved_new += 1
        except ValueError as error:
            flash(f"{part}: {error}", "warning")
        except Exception as error:
            flash(f"{part}: failed to save ({error})", "error")

    if saved_new > 0:
        core.remember_upload_sha(db, customer_id, payload["upload_sha"], kind="outfit_upload", filename=payload.get("upload_name", "outfit"))
        flash(f"Saved {saved_new}/3 garments to wardrobe.", "success")
    if skipped:
        flash("Skipped similar parts: " + ", ".join(skipped), "info")
    if saved_new == 0 and not skipped:
        flash("Nothing new saved.", "info")

    core.cleanup_pending_outfit_payload(payload)
    session.pop("pending_outfit_extract", None)


def action_upload_single(user: dict[str, Any]) -> None:
    file = request.files.get("upload_garment")
    if file is None or file.filename is None or file.filename == "":
        flash("Please upload a garment image.", "error")
        return

    img_bytes = file.read()
    if not img_bytes:
        flash("Uploaded garment is empty.", "error")
        return

    db, fs = core.get_db_fs()
    customer_id = str(user["_id"])
    upload_sha = core.compute_upload_sha256(img_bytes)

    existing_payload = session.get("pending_single_upload")
    if existing_payload and existing_payload.get("upload_sha") != upload_sha:
        core.cleanup_pending_single_payload(existing_payload)
        session.pop("pending_single_upload", None)

    if core.upload_already_used(db, customer_id, upload_sha):
        core.cleanup_pending_single_payload(session.get("pending_single_upload"))
        session.pop("pending_single_upload", None)
        flash("Duplicate upload: this exact image was already uploaded.", "error")
        return

    try:
        part_guess, emb, save_source_img = core.process_single_upload(img_bytes, customer_id)
    except RuntimeError as error:
        flash(str(error), "error")
        return

    if part_guess not in core.PART_ORDER:
        part_guess = "shirt"
        flash("Could not infer type confidently. Defaulting to shirt.", "warning")

    tags = request.form.getlist("tags_manual")
    tags = [tag for tag in tags if tag in core.TAG_OPTIONS]
    auto_style = request.form.get("auto_style_single") == "on"
    if auto_style and not tags:
        guessed = core.infer_tag_from_existing(customer_id, part_guess, emb)
        if guessed:
            tags = [guessed]
            flash(f"Auto style: {guessed}", "info")

    sim_doc, similarity = core.find_most_similar_garment(customer_id, part_guess, emb)
    if sim_doc is not None and similarity >= core.GARMENT_SIMILARITY_WARN_THRESHOLD:
        token = core.make_pending_token("single")
        core.cleanup_pending_single_payload(session.get("pending_single_upload"))
        session["pending_single_upload"] = {
            "upload_sha": upload_sha,
            "upload_name": file.filename,
            "part_guess": part_guess,
            "tags_final": tags,
            "img_path": core.save_pending_image(token, "img", save_source_img),
            "emb_path": core.save_pending_embedding(token, "emb", emb),
            "similar_hit": {"garment_id": str(sim_doc["_id"]), "similarity": float(similarity)},
        }
        flash(f"Potential duplicate detected for {part_guess} ({similarity * 100.0:.2f}% cosine similarity).", "warning")
        return

    try:
        core.save_garment(db, fs, customer_id, part_guess, save_source_img, emb, tags, source="manual_auto")
        core.remember_upload_sha(db, customer_id, upload_sha, kind="single_garment_upload", filename=file.filename)
        flash("Saved garment to wardrobe.", "success")
    except ValueError as error:
        flash(str(error), "warning")

    core.cleanup_pending_single_payload(session.get("pending_single_upload"))
    session.pop("pending_single_upload", None)


def action_confirm_single_review(user: dict[str, Any]) -> None:
    payload = session.get("pending_single_upload")
    if not payload:
        flash("No pending single-garment review found.", "error")
        return

    if request.form.get("save_anyway_single") != "on":
        flash("Skipped saving because confirmation was not checked.", "info")
        core.cleanup_pending_single_payload(payload)
        session.pop("pending_single_upload", None)
        return

    db, fs = core.get_db_fs()
    customer_id = str(user["_id"])
    try:
        image = core.load_rgba_from_path(payload["img_path"])
        emb = np.load(payload["emb_path"], allow_pickle=False).astype(np.float32, copy=False)
        core.save_garment(
            db,
            fs,
            customer_id,
            payload["part_guess"],
            image,
            emb,
            list(payload.get("tags_final", [])),
            source="manual_auto",
        )
        core.remember_upload_sha(db, customer_id, payload["upload_sha"], kind="single_garment_upload", filename=payload.get("upload_name", "single"))
        flash("Saved garment to wardrobe.", "success")
    except ValueError as error:
        flash(str(error), "warning")
    except Exception as error:
        flash(f"Failed to save reviewed garment: {error}", "error")

    core.cleanup_pending_single_payload(payload)
    session.pop("pending_single_upload", None)


def action_run_match1(user: dict[str, Any], filters: dict[str, Any]) -> None:
    start_part = request.form.get("start_part", "")
    start_garment_id = request.form.get("start_garment_id", "")
    cand_each = parse_int(request.form.get("cand_each"), 80, 20, 200)
    try:
        results, message = core.run_match_one(
            str(user["_id"]),
            start_part,
            start_garment_id,
            filters["tags_filter"],
            cand_each,
            filters["threshold"],
            filters["top_k"],
        )
    except Exception as error:
        flash(f"Match failed: {error}", "error")
        session["match1_results"] = []
        return
    session["match1_results"] = results
    if message:
        flash(message, "info")
    else:
        flash(f"Found {len(results)} matches.", "success")


def action_run_match2(user: dict[str, Any], filters: dict[str, Any]) -> None:
    part_a = request.form.get("part_a", "")
    part_b = request.form.get("part_b", "")
    garment_a_id = request.form.get("garment_a_id", "")
    garment_b_id = request.form.get("garment_b_id", "")
    cand_each = parse_int(request.form.get("cand_each2"), 120, 20, 300)
    try:
        results, message = core.run_match_two(
            str(user["_id"]),
            part_a,
            garment_a_id,
            part_b,
            garment_b_id,
            filters["tags_filter"],
            cand_each,
            filters["threshold"],
            filters["top_k"],
        )
    except Exception as error:
        flash(f"Match failed: {error}", "error")
        session["match2_results"] = []
        return
    session["match2_results"] = results
    if message:
        flash(message, "info")
    else:
        flash(f"Found {len(results)} matches.", "success")


def action_run_recommend(user: dict[str, Any], filters: dict[str, Any]) -> None:
    samples = parse_int(request.form.get("samples"), 5000, 200, 100000)
    max_outfits = parse_int(request.form.get("max_outfits"), 20, 1, 500)
    try:
        results, message = core.run_recommendations(
            str(user["_id"]),
            filters["tags_filter"],
            samples,
            max_outfits,
            filters["threshold"],
        )
    except Exception as error:
        flash(f"Recommend failed: {error}", "error")
        session["rec_results"] = []
        return
    session["rec_results"] = results
    if message:
        flash(message, "info")
    else:
        flash(f"Found {len(results)} outfits.", "success")


def action_save_result(user: dict[str, Any], result_key: str, source: str) -> None:
    result_list = session.get(result_key)
    if not isinstance(result_list, list):
        flash("No results to save from.", "error")
        return

    index = parse_int(request.form.get("result_index"), -1)
    if index < 0 or index >= len(result_list):
        flash("Invalid selection to save.", "error")
        return

    row = result_list[index]
    docs = core.get_garments_by_ids([row["shirt_id"], row["pants_id"], row["shoes_id"]])
    shirt_doc = docs.get(row["shirt_id"])
    pants_doc = docs.get(row["pants_id"])
    shoes_doc = docs.get(row["shoes_id"])
    if not (shirt_doc and pants_doc and shoes_doc):
        flash("One or more garments are missing.", "error")
        return

    filters = get_filter_state()
    ok, message = core.save_outfit(
        str(user["_id"]),
        float(row["score"]),
        shirt_doc,
        pants_doc,
        shoes_doc,
        filters["tags_filter"],
        source=source,
    )
    if ok:
        flash("Outfit saved.", "success")
    else:
        flash(message or "Failed to save outfit.", "info")


@app.route("/app", methods=["GET", "POST"])
def app_page():
    tab = request.args.get("tab", "add")
    filters = get_filter_state()

    if request.method == "POST":
        action = request.form.get("action", "")

        if handle_auth_post(action):
            return redirect(url_for("app_page", tab="add"))

        user = get_auth_user()
        if user is None:
            flash("Login is required.", "error")
            return redirect(url_for("app_page", tab="add"))

        if action == "update_filters":
            set_filter_state_from_request()
            return redirect(url_for("app_page", tab=request.form.get("tab", "add")))

        filters = get_filter_state()

        if action == "logout":
            core.cleanup_pending_outfit_payload(session.get("pending_outfit_extract"))
            core.cleanup_pending_single_payload(session.get("pending_single_upload"))
            session.clear()
            flash("Logged out.", "success")
            return redirect(url_for("app_page", tab="add"))

        if action == "upload_outfit":
            action_upload_outfit(user, filters)
            tab = "add"
        elif action == "confirm_outfit_review":
            action_confirm_outfit_review(user)
            tab = "add"
        elif action == "upload_single":
            action_upload_single(user)
            tab = "add"
        elif action == "confirm_single_review":
            action_confirm_single_review(user)
            tab = "add"
        elif action == "skip_single_review":
            payload = session.get("pending_single_upload")
            core.cleanup_pending_single_payload(payload)
            session.pop("pending_single_upload", None)
            flash("Skipped similar garment.", "info")
            tab = "add"
        elif action == "run_match1":
            action_run_match1(user, filters)
            tab = "match1"
        elif action == "save_match1":
            action_save_result(user, "match1_results", "match1")
            tab = "match1"
        elif action == "run_match2":
            action_run_match2(user, filters)
            tab = "match2"
        elif action == "save_match2":
            action_save_result(user, "match2_results", "match2")
            tab = "match2"
        elif action == "run_recommend":
            action_run_recommend(user, filters)
            tab = "recommend"
        elif action == "save_recommend":
            action_save_result(user, "rec_results", "recommend")
            tab = "recommend"
        elif action == "delete_saved_outfit":
            outfit_id = request.form.get("outfit_id", "")
            if core.delete_outfit(str(user["_id"]), outfit_id):
                flash("Deleted successfully.", "success")
            else:
                flash("Item was already deleted.", "info")
            tab = "saved"

        return redirect(url_for("app_page", tab=tab))

    user = get_auth_user()
    if user is None:
        return render_template("auth.html", active_nav="app", user=None, legal_consent_text=core.LEGAL_CONSENT_TEXT)

    filters = get_filter_state()
    customer_id = str(user["_id"])

    wardrobes = {
        part: core.load_wardrobe(customer_id, part, filters["tags_filter"], limit=400)
        for part in core.PART_ORDER
    }

    latest_added_cards: list[dict[str, Any]] = []
    for part in core.PART_ORDER:
        for item in wardrobes.get(part, [])[:6]:
            image_fs_id = item.get("image_fs_id")
            if not image_fs_id:
                continue
            latest_added_cards.append(
                {
                    "part": part,
                    "image_fs_id": str(image_fs_id),
                    "tags": item.get("tags", []) or [],
                    "created_at": item.get("created_at"),
                }
            )
    latest_added_cards.sort(key=lambda row: row.get("created_at") or 0, reverse=True)
    latest_added_cards = latest_added_cards[:9]

    match1_results = session.get("match1_results") if isinstance(session.get("match1_results"), list) else []
    match2_results = session.get("match2_results") if isinstance(session.get("match2_results"), list) else []
    rec_results = session.get("rec_results") if isinstance(session.get("rec_results"), list) else []

    # Saved outfits filters and pagination
    min_saved_score_pct = parse_int(request.args.get("saved_min_score", 0), 0, 0, 100)
    saved_style_filter = [tag for tag in request.args.getlist("saved_style") if tag in core.TAG_OPTIONS]
    saved_page_size = parse_int(request.args.get("saved_page_size", 6), 6, 1, 100)
    saved_page = parse_int(request.args.get("saved_page", 1), 1, 1, 9999)

    saved_filtered = core.list_saved_outfits(customer_id, min_saved_score_pct / 100.0, saved_style_filter)
    total_saved = len(saved_filtered)
    total_pages = max(1, math.ceil(total_saved / max(1, saved_page_size)))
    saved_page = min(saved_page, total_pages)

    start_idx = (saved_page - 1) * saved_page_size
    end_idx = min(total_saved, start_idx + saved_page_size)
    saved_rows = saved_filtered[start_idx:end_idx]

    saved_cards = []
    for row in saved_rows:
        docs = core.get_garments_by_ids([row.get("shirt_id"), row.get("pants_id"), row.get("shoes_id")])
        shirt_doc = docs.get(str(row.get("shirt_id")))
        pants_doc = docs.get(str(row.get("pants_id")))
        shoes_doc = docs.get(str(row.get("shoes_id")))
        if not (shirt_doc and pants_doc and shoes_doc):
            continue

        saved_cards.append(
            {
                "id": str(row.get("_id")),
                "score": float(row.get("score", 0.0)),
                "score_label": core.fmt_score_100(float(row.get("score", 0.0))),
                "shirt_image": shirt_doc.get("image_fs_id"),
                "pants_image": pants_doc.get("image_fs_id"),
                "shoes_image": shoes_doc.get("image_fs_id"),
                "tags": row.get("tags", []),
                "source": row.get("source"),
                "created_at": row.get("created_at"),
            }
        )

    # Pending review previews
    pending_outfit = session.get("pending_outfit_extract")
    pending_single = session.get("pending_single_upload")

    pending_outfit_image = pending_outfit_preview(pending_outfit)
    pending_single_image = pending_single_preview(pending_single)

    pending_outfit_similar = {}
    if pending_outfit:
        for part, info in (pending_outfit.get("similar_hits") or {}).items():
            garment = core.get_garment_by_id(info.get("garment_id"))
            pending_outfit_similar[part] = {
                "similarity_pct": float(info.get("similarity", 0.0)) * 100.0,
                "image_fs_id": garment.get("image_fs_id") if garment else None,
            }

    pending_single_similar = None
    if pending_single:
        hit = pending_single.get("similar_hit") or {}
        garment = core.get_garment_by_id(hit.get("garment_id")) if hit else None
        pending_single_similar = {
            "similarity_pct": float(hit.get("similarity", 0.0)) * 100.0,
            "image_fs_id": garment.get("image_fs_id") if garment else None,
            "part_guess": pending_single.get("part_guess"),
        }

    return render_template(
        "app.html",
        active_nav="app",
        user=user,
        tab=tab,
        filters=filters,
        wardrobes=wardrobes,
        match1_cards=format_results_for_display(match1_results),
        match2_cards=format_results_for_display(match2_results),
        rec_cards=format_results_for_display(rec_results),
        latest_added_cards=latest_added_cards,
        pending_outfit=pending_outfit,
        pending_outfit_image=pending_outfit_image,
        pending_outfit_similar=pending_outfit_similar,
        pending_single=pending_single,
        pending_single_image=pending_single_image,
        pending_single_similar=pending_single_similar,
        saved_cards=saved_cards,
        saved_total=total_saved,
        saved_start=start_idx + 1 if total_saved else 0,
        saved_end=end_idx,
        saved_page=saved_page,
        saved_page_size=saved_page_size,
        saved_total_pages=total_pages,
        saved_min_score=min_saved_score_pct,
        saved_style_filter=saved_style_filter,
        inference_status=core.get_inference_status(),
    )


@app.route("/delete-garments", methods=["GET", "POST"])
@require_auth
def delete_garments_page():
    user = get_auth_user()
    assert user is not None
    customer_id = str(user["_id"])

    if request.method == "POST":
        garment_id = request.form.get("garment_id", "")
        garment_doc = core.get_garment_by_id(garment_id)
        if not garment_doc or str(garment_doc.get("customer_id")) != customer_id:
            flash("Garment not found.", "error")
        else:
            deleted_outfits = core.delete_garment_and_related_outfits(customer_id, garment_doc)
            flash(f"Garment deleted. Related outfits removed: {deleted_outfits}.", "success")
        return redirect(url_for("delete_garments_page", **request.args.to_dict(flat=True)))

    part_filter = request.args.get("part", "all")
    if part_filter not in ("all", "shirt", "pants", "shoes"):
        part_filter = "all"

    page_size = parse_int(request.args.get("page_size", 20), 20, 10, 200)
    page_num = parse_int(request.args.get("page", 1), 1, 1, 9999)

    db, _ = core.get_db_fs()
    query: dict[str, Any] = {"customer_id": customer_id}
    if part_filter != "all":
        query["part"] = part_filter

    total = int(db["Wardrobe"].count_documents(query))
    total_pages = max(1, math.ceil(total / max(1, page_size)))
    page_num = min(page_num, total_pages)
    skip = (page_num - 1) * page_size

    docs = list(
        db["Wardrobe"]
        .find(query, {"_id": 1, "part": 1, "tags": 1, "created_at": 1, "image_fs_id": 1})
        .sort("created_at", -1)
        .skip(skip)
        .limit(page_size)
    )

    garment_refs = [(str(doc.get("part")), str(doc.get("_id"))) for doc in docs]
    related_counts = core.get_related_outfit_counts(customer_id, garment_refs)

    cards = []
    for doc in docs:
        cards.append(
            {
                "id": str(doc.get("_id")),
                "part": doc.get("part"),
                "tags": doc.get("tags", []),
                "created_at": doc.get("created_at"),
                "image_fs_id": doc.get("image_fs_id"),
                "related_outfits": related_counts.get((str(doc.get("part")), str(doc.get("_id"))), 0),
            }
        )

    start_idx = skip + 1 if total > 0 else 0
    end_idx = min(total, skip + page_size)

    return render_template(
        "delete_garments.html",
        active_nav="app",
        user=user,
        cards=cards,
        part_filter=part_filter,
        page_size=page_size,
        page_num=page_num,
        total_pages=total_pages,
        total=total,
        start_idx=start_idx,
        end_idx=end_idx,
    )


if __name__ == "__main__":  # pragma: no cover
    app.run(host="127.0.0.1", port=8000, debug=True)
