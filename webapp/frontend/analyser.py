"""
Streamlit Frontend — Mental Health Detection
Two modes, selected by the MINDSCAN_API environment variable:
  set   (e.g. http://localhost:8000) → call the FastAPI backend (Docker/local)
  unset → run the model in-process (Streamlit Community Cloud, single process)
Run: streamlit run webapp/frontend/app.py
"""

import base64
import os

import requests
import streamlit as st
from spellchecker import SpellChecker

import shared

_spell = SpellChecker()

API = os.environ.get("MINDSCAN_API")

if API:
    def run_predict(text):
        r = requests.post(f"{API}/predict", json={"text": text}, timeout=90)
        r.raise_for_status()
        return r.json()

    def run_lime(text):
        r = requests.post(f"{API}/explain/lime", json={"text": text}, timeout=300)
        r.raise_for_status()
        return r.json()

    def run_shap(text):
        r = requests.post(f"{API}/explain/shap", json={"text": text}, timeout=300)
        r.raise_for_status()
        return r.json()
else:
    import inference
    run_predict = inference.predict
    run_lime    = inference.explain_lime
    run_shap    = inference.explain_shap

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

EXAMPLES = {
    "Anxiety/Stress": "It's been 3 weeks I feel really tired. The heart is pounding, the body is shaky. In this brain it doesn't stop thinking about things. The feeling of not being needed, the feeling of being ignored. The soul is restless. Nervous. Afraid. I cannot concentrate at work and keep worrying about everything.",
    "Bipolar":        "CBT? DBT? Anyone had good experiences? I'm running out of drugs I can try. Vibryd, Lamictal, Latuda, and welbutrin worked for two years but then I crashed and i'm back to zero. Have tried many drugs but pdoc is useless for therapy otherwise. Want to try other therapy stuff in concert with drugs. Does anything work on bipolar?",
    "Depression":     "I have been told that I should take anti-depressants, but they zombify me. I am already numb, what I need is to feel alive again. I am sick of waking up empty everyday. Nothing brings me joy anymore and I just go through the motions. How do you deal with your depression?",
    "Suicidal":       "Going to end it tonight. I am tired of being in this endless loop, dying day after day. I know it is pretty pointless but the hurt is just too much to bear. Nobody would notice anyway. I have written the notes. It ends tonight.",
}


def validate_input(text: str):
    words = [w.strip("'\".,!?;:") for w in text.split()]
    real  = [w for w in words if w.isalpha() and len(w) >= 3]
    if not real:
        return False, "short", 0, 0
    unknown = _spell.unknown(real)
    known   = len(real) - len(unknown)
    ratio   = known / len(real)
    unique  = len(set(w.lower() for w in real))
    if len(real) < 10:
        return False, "short", len(real), ratio
    if ratio < 0.5:
        return False, "gibberish", len(real), ratio
    if unique < 5:
        return False, "repeated", len(real), ratio
    return True, "ok", len(real), ratio


shared.inject_css()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>Mental Health Detection from Social Media Text</h1>
    <p>Paste or type a social media post to classify the expressed mental health state.<br>
       Predictions are explained using <strong>LIME</strong> and <strong>SHAP</strong> feature attribution.</p>
    <span class="badge">BERT Fine-tuned · 5 Classes · F1 0.841</span>
</div>
""", unsafe_allow_html=True)


# ── Backend check (API mode only) ─────────────────────────────────────────────
@st.cache_data(ttl=30)
def check_backend():
    try:
        return requests.get(f"{API}/", timeout=5).status_code == 200
    except Exception:
        return False

if API and not check_backend():
    st.error("**Backend offline.** Start it with: `cd webapp/backend && uvicorn main:app`")
    st.stop()


# ── Input section ─────────────────────────────────────────────────────────────
st.markdown('<div class="card-title">Try a sample</div>', unsafe_allow_html=True)

# Example buttons — session state must be set BEFORE text_area renders
btn_cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(btn_cols, EXAMPLES.items()):
    icon = CLASS_ICONS.get(label, '')
    if col.button(f"{icon} {label}", use_container_width=True, key=f"btn_{label}"):
        st.session_state['text_area'] = text

col_clear, _ = st.columns([1, 9])
with col_clear:
    if st.button("Clear", key="btn_clear"):
        st.session_state['text_area'] = ''

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="card-title">Your text</div>', unsafe_allow_html=True)

text_input = st.text_area(
    label="",
    height=150,
    placeholder="Paste or type a social media post here (min. 10 real words)…",
    key="text_area",
    label_visibility="collapsed",
)

# ── Validation feedback ───────────────────────────────────────────────────────
enough_text = False
if text_input.strip():
    valid, reason, word_count, ratio = validate_input(text_input)
    enough_text = valid

    if reason == "short":
        needed = 10 - word_count
        st.markdown(
            f'<div class="word-counter warn">{word_count}/10 words — add {needed} more to continue</div>',
            unsafe_allow_html=True,
        )
    elif reason == "gibberish":
        st.warning("Input contains too many unrecognised words. Please enter valid English text.")
    elif reason == "repeated":
        st.warning("Input appears to be repeated words. Please enter meaningful sentences.")
    else:
        st.markdown(
            f'<div class="word-counter ready">✓ {word_count} words · {ratio:.0%} valid English</div>',
            unsafe_allow_html=True,
        )

analyse = st.button("Analyse Text", type="primary", disabled=not enough_text)


# ── Results ───────────────────────────────────────────────────────────────────
if analyse and text_input.strip():
    with st.spinner("Running prediction…"):
        try:
            result = run_predict(text_input)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    prediction = result['prediction']
    confidence = result['confidence']
    probs      = result['probabilities']
    colour     = CLASS_COLOURS.get(prediction, '#4B5563')
    icon       = CLASS_ICONS.get(prediction, '')

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    col_result, col_dist = st.columns([2, 3], gap="large")

    with col_result:
        st.markdown('<div class="card-title">Prediction</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="result-card" style="background:{colour}">'
            f'<div class="result-label">Detected condition</div>'
            f'<div class="result-class">{icon} {prediction}</div>'
            f'<div class="result-conf">Confidence &nbsp;<strong>{confidence:.1%}</strong></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if confidence < 0.5:
            st.warning(f"Low confidence ({confidence:.1%}) — model is uncertain. Review the distribution.")

    with col_dist:
        st.markdown('<div class="card-title">Class probabilities</div>', unsafe_allow_html=True)
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        for i, (cls, prob) in enumerate(sorted_probs):
            bar_colour = CLASS_COLOURS.get(cls, '#6B7280')
            row_class  = "prob-row prob-row-top" if i == 0 else "prob-row"
            bar_width  = f"{prob * 100:.1f}%"
            st.markdown(
                f'<div class="{row_class}">'
                f'<span class="prob-label">{CLASS_ICONS.get(cls,"")} {cls}</span>'
                f'<div class="prob-bar-wrap"><div class="prob-bar" style="width:{bar_width};background:{bar_colour};opacity:0.85"></div></div>'
                f'<span class="prob-pct">{prob:.1%}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Explanations ──────────────────────────────────────────────────────────
    st.markdown('<div class="card-title">Explainability</div>', unsafe_allow_html=True)
    tab_lime, tab_shap = st.tabs(["LIME Feature Attribution", "SHAP Feature Attribution"])

    with tab_lime:
        with st.spinner("Generating LIME explanation… (can take a minute on CPU-only machines)"):
            try:
                st.image(base64.b64decode(run_lime(text_input)['image']), use_container_width=True)
                st.caption("Green bars = words that support the predicted class  ·  Red bars = words pushing against it.")
            except Exception as e:
                st.error(f"LIME explanation failed: {e}")

    with tab_shap:
        with st.spinner("Generating SHAP explanation… (can take a minute on CPU-only machines)"):
            try:
                st.image(base64.b64decode(run_shap(text_input)['image']), use_container_width=True)
                st.caption("Green bars = words push probability toward this class  ·  Red bars = words push probability away.")
            except Exception as e:
                st.error(f"SHAP explanation failed: {e}")

    shared.explainer_expander()


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    ⚠️ <strong>Research prototype only.</strong>
    This tool is not a clinical diagnostic instrument and must not be used as a substitute
    for professional mental health assessment. Intended for academic research purposes only.<br>
    COM748 Masters Research Project · Fine-tuned BERT · LIME &amp; SHAP Explainability
</div>
""", unsafe_allow_html=True)
