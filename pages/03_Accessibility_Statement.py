import streamlit as st

from brand_theme import inject_glass_css, render_footer, render_top_nav

BRAND = "VibeCheck"
CONTACT_EMAIL = "eitanmamedov7@gmail.com"
LAST_REVIEW = "2026-02-28"

st.set_page_config(page_title=f"{BRAND} Accessibility", layout="wide")
inject_glass_css(hide_sidebar=True)
render_top_nav(active="legal")

st.markdown(
    f"""
<div class="landing-shell page-shell">
  <div class="glass-panel hero-panel">
    <p class="eyebrow">Legal</p>
    <h1 style="margin:10px 0 8px 0; font-size:1.7rem;">Accessibility Statement</h1>
    <p class="hero-sub">Last reviewed: {LAST_REVIEW}</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="page-shell card legal-card">
{BRAND} aims to conform, as far as practical, with Israeli Standard SI 5568 based on WCAG 2.0 Level AA.

### Accessibility Measures
- Keyboard navigation for core flows
- Clear semantic heading structure
- Sufficient text contrast and readable typography
- Accessible labels/alt text where relevant
- Compatibility with common modern browsers and assistive technologies

### Known Limitations
As a beta service, some areas may still need accessibility improvements.

### Contact
For accessibility issues, contact: {CONTACT_EMAIL}
</div>
""",
    unsafe_allow_html=True,
)

render_footer()
