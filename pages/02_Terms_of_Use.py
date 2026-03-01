import streamlit as st

from brand_theme import inject_glass_css, render_footer, render_top_nav

BRAND = "VibeCheck"
CONTACT_EMAIL = "eitanmamedov7@gmail.com"
LAST_UPDATED = "2026-02-28"
TERMS_VERSION = "2026-02-28"

st.set_page_config(page_title=f"{BRAND} Terms of Use", layout="wide")
inject_glass_css(hide_sidebar=True)
render_top_nav(active="legal")

st.markdown(
    f"""
<div class="landing-shell page-shell">
  <div class="glass-panel hero-panel">
    <p class="eyebrow">Legal</p>
    <h1 style="margin:10px 0 8px 0; font-size:1.7rem;">Terms of Use</h1>
    <p class="hero-sub">Effective date: {LAST_UPDATED} | Terms version: {TERMS_VERSION}</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="page-shell card legal-card">
### 1) Acceptance
By creating an account or using {BRAND}, you agree to these Terms and the Privacy Policy.

### 2) Beta / Private Preview
{BRAND} is experimental, may change or go offline, and is provided **"as is"** and **"as available"**.
Outputs may be inaccurate.

### 3) No Warranties and Limited Liability
To the maximum extent permitted by law, warranties are disclaimed and liability is limited.
We are not responsible for decisions based on scores/recommendations.

### 4) User Responsibilities
You must only upload images you own or have rights to use.
No illegal content, abuse, or attempts to break security.

### 5) IP and User Content
{BRAND} owns the software and service IP.
Users retain rights in uploads but grant {BRAND} a limited license to host/process content to provide the service.

### 6) Account Termination
We may suspend abusive accounts.
Users may stop using the service and request deletion.

### 7) Governing Law
These Terms are governed by laws of Israel, with venue in competent courts in Israel.

### 8) Changes
We may update Terms and may require re-acceptance.
Current version: {TERMS_VERSION}.

### 9) Contact
{CONTACT_EMAIL}
</div>
""",
    unsafe_allow_html=True,
)

render_footer()
