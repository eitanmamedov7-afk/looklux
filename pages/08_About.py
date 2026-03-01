import streamlit as st

from brand_theme import BRAND_NAME, inject_glass_css, render_footer, render_top_nav

st.set_page_config(page_title=f"{BRAND_NAME} | About", layout="wide")
inject_glass_css(hide_sidebar=True)
render_top_nav(active="about")

st.markdown(
    f"""
<div class="landing-shell page-shell">
  <div class="glass-panel hero-panel">
    <p class="eyebrow">About {BRAND_NAME}</p>
    <h1 class="hero-title">Focused Intelligence, Premium Interaction.</h1>
    <p class="hero-sub" style="max-width:72ch;">
      {BRAND_NAME} is built for teams and individuals who need visual analysis with clear scoring logic.
      The platform combines robust model execution with a calm, trust-focused interface.
    </p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="page-shell section-tight">
  <div class="section-grid">
    <article class="glass-panel section-panel">
      <h4>Principle 01</h4>
      <p>Clarity first: every step of processing should be understandable and reviewable.</p>
    </article>
    <article class="glass-panel section-panel">
      <h4>Principle 02</h4>
      <p>Signal over noise: compact interfaces with strong visual hierarchy.</p>
    </article>
    <article class="glass-panel section-panel">
      <h4>Principle 03</h4>
      <p>Responsible speed: fast workflows with confidence-aware outputs.</p>
    </article>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

render_footer()
