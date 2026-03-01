import streamlit as st

from brand_theme import inject_glass_css, render_footer, render_top_nav

BRAND = "VibeCheck"
CONTACT_EMAIL = "eitanmamedov7@gmail.com"
LAST_UPDATED = "2026-02-28"

st.set_page_config(page_title=f"{BRAND} Privacy Policy", layout="wide")
inject_glass_css(hide_sidebar=True)
render_top_nav(active="legal")

st.markdown(
    f"""
<div class="landing-shell page-shell">
  <div class="glass-panel hero-panel">
    <p class="eyebrow">Legal</p>
    <h1 style="margin:10px 0 8px 0; font-size:1.7rem;">Privacy Policy</h1>
    <p class="hero-sub">Effective date: {LAST_UPDATED} | Last updated: {LAST_UPDATED}</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="page-shell card legal-card">
### 1) Who We Are
{BRAND} is a private preview / beta wardrobe and outfit scoring service currently in testing.  
Contact: {CONTACT_EMAIL}

### 2) Beta Context
The service is experimental and may be changed, interrupted, or removed at any time.  
{BRAND} is not a bank, medical provider, or legal advisor.

### 3) Data We Collect
- **Account data**: name, email, password hash (bcrypt), and account creation timestamp (`created_at`).
- **Wardrobe and outfit data**:
  - Uploaded images stored in MongoDB GridFS.
  - Garment images derived from uploaded outfits (shirt, pants, shoes crops).
  - Embeddings (vectors) stored as binary float16 values.
  - Style tags selected by users.
  - SHA-256 image hashes for duplicate detection.
  - Outfit combinations (`shirt_id`, `pants_id`, `shoes_id`), scores, tags, source, and timestamps.
- **Operational data**: basic runtime logs/errors from Streamlit/runtime and backend services.
- **Session/local state data**: essential session behavior required to operate the app.

### 4) Why We Collect Data
Authentication, wardrobe storage, compatibility scoring, duplicate prevention, abuse reduction, and beta diagnostics.

### 5) Legal Basis and Consent
By creating an account and using {BRAND}, you consent to processing required to provide the service.  
Users must accept Terms + Privacy before account creation.

### 6) Sharing
We do **not** sell personal data.  
We may share with service providers (hosting / MongoDB infrastructure) only as needed, or if required by law.

### 7) Retention
Account and wardrobe data are retained until deletion request/account removal.  
Diagnostic logs are retained for limited periods (typically up to 90 days), unless longer retention is legally required.

### 8) Security
Reasonable safeguards include bcrypt password hashing, access controls, and standard security practices.  
No system is perfectly secure.

### 9) User Rights
You may request access, correction, or deletion by contacting {CONTACT_EMAIL}.  
We respond within a reasonable time.

### 10) Children
{BRAND} is not intended for children under 13.

### 11) International Transfers
Data may be processed outside Israel depending on infrastructure locations, with reasonable safeguards.

### 12) Policy Changes
We may update this policy and may require acceptance of updated versions.
</div>
""",
    unsafe_allow_html=True,
)

render_footer()
