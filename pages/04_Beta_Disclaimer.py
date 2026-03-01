import streamlit as st

from brand_theme import inject_glass_css, render_footer, render_top_nav

BRAND = "VibeCheck"
CONTACT_EMAIL = "eitanmamedov7@gmail.com"
LAST_UPDATED = "2026-02-28"

st.set_page_config(page_title=f"{BRAND} Beta Disclaimer", layout="wide")
inject_glass_css(hide_sidebar=True)
render_top_nav(active="legal")

st.markdown(
    f"""
<div class="landing-shell page-shell">
  <div class="glass-panel hero-panel">
    <p class="eyebrow">Legal</p>
    <h1 style="margin:10px 0 8px 0; font-size:1.7rem;">Beta Disclaimer / Private Preview Notice</h1>
    <p class="hero-sub">Effective date: {LAST_UPDATED}</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="page-shell card legal-card">
{BRAND} is currently in a limited beta / private preview.

- The service is experimental and may be inaccurate.
- Provided **"as is"** and **"as available"**.
- We do not provide professional advice.
- We are not responsible for decisions based on outputs.
- To the maximum extent permitted by law, warranties are disclaimed and liability is limited.

Access is available during beta for users who accept Terms and Privacy.

Contact: {CONTACT_EMAIL}
</div>
""",
    unsafe_allow_html=True,
)

render_footer()
