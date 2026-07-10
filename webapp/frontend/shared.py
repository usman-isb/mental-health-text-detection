"""Shared styling, paths and helpers for all Streamlit pages."""

from pathlib import Path

# repo layout: data/webapp/frontend/shared.py → data/
DATA_DIR    = Path(__file__).resolve().parents[2]
RESULTS_DIR = DATA_DIR / 'results'
CHARTS_DIR  = RESULTS_DIR / 'charts'

CLASS_COLOURS = {
    'Anxiety/Stress': '#D97706',
    'Bipolar':        '#7C3AED',
    'Depression':     '#2563EB',
    'Normal':         '#059669',
    'Suicidal':       '#9F1239',
}

CLASS_ICONS = {
    'Anxiety/Stress': '😰',
    'Bipolar':        '🔄',
    'Depression':     '😔',
    'Normal':         '😊',
    'Suicidal':       '🆘',
}

CSS = """
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }

/* ── Header banner ── */
.app-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 28px;
    color: white;
}
.app-header h1 { margin: 0; font-size: 1.65rem; font-weight: 700; letter-spacing: -0.3px; }
.app-header p  { margin: 6px 0 0; font-size: 0.88rem; opacity: 0.82; line-height: 1.5; }
.badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.75rem;
    margin-top: 10px;
    letter-spacing: 0.4px;
}

/* ── Section cards ── */
.card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 22px 26px;
    margin-bottom: 18px;
}
.card-title {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #6b7280;
    margin-bottom: 14px;
}

/* ── Example pill buttons ── */
.stButton > button {
    border-radius: 6px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 6px 10px !important;
    border: 1px solid #d1d5db !important;
    background: #f9fafb !important;
    color: #374151 !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: #f3f4f6 !important;
    border-color: #9ca3af !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: #1e3a5f !important;
    color: white !important;
    border: none !important;
    font-size: 0.9rem !important;
    padding: 10px 28px !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
}
.stButton > button[kind="primary"]:hover {
    background: #2d5382 !important;
}
.stButton > button[kind="primary"]:disabled {
    background: #9ca3af !important;
    cursor: not-allowed !important;
}

/* ── Text area ── */
.stTextArea textarea {
    border-radius: 8px !important;
    border: 1px solid #d1d5db !important;
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
    color: #111827 !important;
}
.stTextArea textarea:focus {
    border-color: #2d6a9f !important;
    box-shadow: 0 0 0 3px rgba(45,106,159,0.12) !important;
}

/* ── Prediction result card ── */
.result-card {
    border-radius: 10px;
    padding: 22px 26px;
    color: white;
    margin-bottom: 16px;
}
.result-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    opacity: 0.85;
    margin-bottom: 4px;
}
.result-class {
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    line-height: 1.1;
}
.result-conf {
    font-size: 0.88rem;
    opacity: 0.88;
    margin-top: 6px;
}

/* ── Probability row ── */
.prob-row {
    display: flex;
    align-items: center;
    margin: 5px 0;
    gap: 10px;
}
.prob-label {
    width: 175px;
    font-size: 0.84rem;
    color: #374151;
    white-space: nowrap;
    flex-shrink: 0;
}
.prob-bar-wrap {
    flex: 1;
    background: #f3f4f6;
    border-radius: 4px;
    height: 10px;
    overflow: hidden;
}
.prob-bar {
    height: 10px;
    border-radius: 4px;
    transition: width 0.3s ease;
}
.prob-pct {
    width: 42px;
    font-size: 0.82rem;
    color: #6b7280;
    text-align: right;
    flex-shrink: 0;
}
.prob-row-top .prob-label { font-weight: 600; color: #111827; }
.prob-row-top .prob-pct   { font-weight: 600; color: #111827; }

/* ── Divider ── */
.section-divider {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 20px 0;
}

/* ── Footer ── */
.app-footer {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 12px 18px;
    font-size: 0.78rem;
    color: #6b7280;
    margin-top: 24px;
    line-height: 1.5;
}

/* ── Warning / info ── */
.stAlert { border-radius: 8px !important; font-size: 0.85rem !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 2px solid #e5e7eb;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #6b7280 !important;
    padding: 8px 18px !important;
    border-radius: 6px 6px 0 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #1e3a5f !important;
    border-bottom: 2px solid #1e3a5f !important;
    background: transparent !important;
}

/* word counter */
.word-counter {
    font-size: 0.78rem;
    color: #9ca3af;
    text-align: right;
    margin-top: -8px;
    margin-bottom: 8px;
}
.word-counter.ready  { color: #059669; font-weight: 500; }
.word-counter.warn   { color: #D97706; }

/* ── Stat tiles ── */
.stat-row { display: flex; gap: 14px; margin-bottom: 18px; flex-wrap: wrap; }
.stat-tile {
    flex: 1;
    min-width: 150px;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 16px 20px;
}
.stat-tile .stat-value { font-size: 1.5rem; font-weight: 700; color: #1e3a5f; letter-spacing: -0.5px; }
.stat-tile .stat-label {
    font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.8px; color: #6b7280; margin-top: 2px;
}
.stat-tile.highlight { background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%); border: none; }
.stat-tile.highlight .stat-value, .stat-tile.highlight .stat-label { color: white; }
.stat-tile.highlight .stat-label { opacity: 0.8; }

/* ── Section heading ── */
.section-head {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e3a5f;
    margin: 26px 0 4px;
    letter-spacing: -0.2px;
}
.section-sub { font-size: 0.84rem; color: #6b7280; margin-bottom: 14px; line-height: 1.55; }

/* ── Sidebar nav ── */
[data-testid="stSidebarNav"] a span { font-size: 0.9rem; }
</style>
"""

FOOTER = """
<div class="app-footer">
    ⚠️ <strong>Research prototype only.</strong>
    This tool is not a clinical diagnostic instrument and must not be used as a substitute
    for professional mental health assessment. Intended for academic research purposes only.<br>
    COM748 Masters Research Project · Fine-tuned BERT · LIME &amp; SHAP Explainability
</div>
"""


def inject_css():
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)


def header(title, subtitle, badge):
    import streamlit as st
    st.markdown(f"""
<div class="app-header">
    <h1>{title}</h1>
    <p>{subtitle}</p>
    <span class="badge">{badge}</span>
</div>
""", unsafe_allow_html=True)


def section(title, sub=None):
    import streamlit as st
    st.markdown(f'<div class="section-head">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="section-sub">{sub}</div>', unsafe_allow_html=True)


def stat_row(stats):
    """stats: list of (value, label) or (value, label, True-for-highlight)."""
    import streamlit as st
    tiles = ''
    for s in stats:
        value, label = s[0], s[1]
        cls = 'stat-tile highlight' if (len(s) > 2 and s[2]) else 'stat-tile'
        tiles += (f'<div class="{cls}"><div class="stat-value">{value}</div>'
                  f'<div class="stat-label">{label}</div></div>')
    st.markdown(f'<div class="stat-row">{tiles}</div>', unsafe_allow_html=True)
