import streamlit as st

BRAND_NAME = "VibeCheck"

_BASE_CSS = """
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
[data-testid="stSidebar"] * {
    color: var(--vc-text);
}
.block-container {
    padding-top: 1.1rem;
    padding-bottom: 1.5rem;
}
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
    margin-bottom: 0.8rem;
    animation: revealSlide .65s ease-out;
}
.landing-nav {
    padding: 12px 14px;
    border-radius: 999px;
    display: flex;
    align-items: center;
    justify-content: space-between;
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
.nav-active {
    display: inline-flex;
    justify-content: center;
    align-items: center;
    text-decoration: none;
    color: var(--vc-text);
    border: 1px solid rgba(255,255,255,0.22);
    background: rgba(11,11,11,0.84);
    padding: 10px 18px;
    min-width: 150px;
    white-space: nowrap;
    border-radius: 999px;
    font-size: 12px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 700;
}
[data-testid="stPageLink"] a {
    display: inline-flex;
    justify-content: center;
    align-items: center;
    text-decoration: none;
    color: var(--vc-muted);
    border: 1px solid rgba(255,255,255,0.14);
    background: rgba(7,7,7,0.72);
    padding: 10px 18px;
    min-width: 150px;
    white-space: nowrap;
    border-radius: 999px;
    font-size: 12px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    transition: transform .2s ease, color .2s ease, border-color .2s ease;
}
[data-testid="stPageLink"] a:hover {
    transform: translateY(-2px);
    color: #fff;
    border-color: rgba(255,255,255,0.24);
}
.hero-panel {
    padding: 24px;
    animation: revealSlide .75s ease-out;
}
.eyebrow, .kicker {
    margin: 0;
    color: var(--vc-muted);
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 11px;
}
.hero-title, .title {
    margin: 10px 0 8px 0;
    line-height: 1.05;
    font-size: clamp(1.85rem, 1.25rem + 1.9vw, 3rem);
    font-weight: 800;
}
.hero-sub, .muted {
    margin: 0;
    color: var(--vc-muted);
    font-size: 1rem;
    line-height: 1.5;
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
.card,
.glass,
.glass-card {
    background: var(--vc-panel);
    border: 1px solid var(--vc-border);
    border-radius: 22px;
    padding: 14px 14px;
    box-shadow: var(--vc-shadow-soft);
    backdrop-filter: blur(14px);
}
.section {
    margin-top: 0.2rem;
}
.section-tight {
    margin-top: 0.8rem;
}
.page-shell {
    max-width: 1080px;
    margin: 0 auto;
}
.legal-card h3 {
    margin: 0.8rem 0 0.5rem 0;
}
.legal-card p, .legal-card li {
    color: var(--vc-muted);
    line-height: 1.6;
}
.site-footer {
    margin-top: 1rem;
    display: flex;
    justify-content: space-between;
    gap: 10px;
    align-items: center;
    font-size: 12px;
    color: var(--vc-muted);
}
.media-frame {
    width: 100%;
    border-radius: 16px;
    border: 1px solid var(--vc-border);
    background: rgba(0,0,0,0.10);
    overflow: hidden;
}
.media-frame img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
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
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
[data-testid="stSlider"] {
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,.16) !important;
    background: rgba(12,12,12,.78) !important;
    color: #f2f2f2 !important;
    backdrop-filter: blur(8px);
}
@keyframes revealSlide {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes auroraShift {
    0% { background-position: 0% 0%, 100% 0%, 0% 0%; }
    100% { background-position: 8% 6%, 92% 4%, 0% 0%; }
}
"""


def inject_glass_css(hide_sidebar: bool = False):
    extra = ""
    if hide_sidebar:
        extra = """
section[data-testid="stSidebar"],
button[kind="header"] {
  display: none !important;
}
"""
    st.markdown(f"<style>{_BASE_CSS}\n{extra}</style>", unsafe_allow_html=True)


def render_top_nav(active: str = "home"):
    st.markdown('<div class="landing-shell"><div class="landing-nav glass-panel">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns([1.6, 1.2, 1.2, 1.2, 1.2, 1.6], vertical_alignment="center")
    is_authenticated = st.session_state.get("auth_user") is not None

    with c1:
        st.markdown(f'<span class="brand-pill">{BRAND_NAME}</span>', unsafe_allow_html=True)

    with c2:
        if active == "home":
            st.markdown('<span class="nav-active">Home</span>', unsafe_allow_html=True)
        else:
            st.page_link("pages/00_Home.py", label="Home")
    with c3:
        if active == "about":
            st.markdown('<span class="nav-active">About</span>', unsafe_allow_html=True)
        else:
            st.page_link("pages/08_About.py", label="About")
    with c4:
        if active == "legal":
            st.markdown('<span class="nav-active">Legal</span>', unsafe_allow_html=True)
        else:
            st.page_link("pages/02_Terms_of_Use.py", label="Legal")
    with c5:
        if active in ("app", "auth"):
            st.markdown('<span class="nav-active">App</span>', unsafe_allow_html=True)
        else:
            st.page_link("wardrobe_app_auth.py", label="App")
    with c6:
        if not is_authenticated:
            st.page_link("wardrobe_app_auth.py", label="LOGIN/GET ACCESS")

    st.markdown("</div></div>", unsafe_allow_html=True)


def render_footer():
    st.markdown(
        """
<div class="page-shell card site-footer">
  <div>VibeCheck</div>
  <div>Visual intelligence platform.</div>
</div>
""",
        unsafe_allow_html=True,
    )

