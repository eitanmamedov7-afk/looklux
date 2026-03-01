import streamlit as st

from brand_theme import BRAND_NAME, inject_glass_css, render_footer, render_top_nav

st.set_page_config(page_title=f"{BRAND_NAME} | Home", layout="wide")
inject_glass_css(hide_sidebar=True)
render_top_nav(active="home")

st.markdown(
    f"""
<div class="landing-shell">
  <div class="glass-panel hero-panel page-shell">
    <p class="eyebrow">Navigation Landing</p>
    <h1 class="hero-title">{BRAND_NAME} Helps You Understand Visual Data With Confidence.</h1>
    <p class="hero-sub">
      Premium model-guided analysis in a clean, high-trust interface. Start here to access the platform,
      authenticate, and review the full inference flow.
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
      <h4>Input Parsing</h4>
      <p>FashnHumanParser segments image regions for clean downstream signals.</p>
    </article>
    <article class="glass-panel section-panel">
      <h4>Feature Engine</h4>
      <p>Frozen ResNet50 embeddings capture stable high-level visual representations.</p>
    </article>
    <article class="glass-panel section-panel">
      <h4>Scoring Head</h4>
      <p>PCA + MLP convert feature vectors into compact compatibility scores.</p>
    </article>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="page-shell section-tight">
  <div class="glass-panel hero-panel">
    <p class="eyebrow">How It Works</p>
    <h2 style="margin:8px 0 14px 0; font-size:1.4rem;">Step-by-Step Inference Flow</h2>
    <div class="section-grid">
      <article class="card">
        <p class="kicker">Step 1</p>
        <h4 style="margin:6px 0 8px 0;">Upload</h4>
        <p class="muted">Users submit image input inside the authenticated workspace.</p>
      </article>
      <article class="card">
        <p class="kicker">Step 2</p>
        <h4 style="margin:6px 0 8px 0;">Segment</h4>
        <p class="muted">Parser isolates garment regions and prepares structured masks.</p>
      </article>
      <article class="card">
        <p class="kicker">Step 3</p>
        <h4 style="margin:6px 0 8px 0;">Embed</h4>
        <p class="muted">ResNet50 generates normalized feature embeddings for each part.</p>
      </article>
      <article class="card">
        <p class="kicker">Step 4</p>
        <h4 style="margin:6px 0 8px 0;">Score</h4>
        <p class="muted">PCA + MLP compute output scores and decision-ready signals.</p>
      </article>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

render_footer()
