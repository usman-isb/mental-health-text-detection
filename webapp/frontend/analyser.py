"""
Streamlit Frontend — Mental Health Detection
Two modes, selected by the MINDSCAN_API environment variable:
  set   (e.g. http://localhost:8000) → call the FastAPI backend (Docker/local)
  unset → run the model in-process (Streamlit Community Cloud, single process)
Run: streamlit run webapp/frontend/app.py
"""

import base64
import html
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
    'Anxiety/Stress': [
        "It's been 3 weeks I feel really tired. The heart is pounding, the body is shaky. In this brain it doesn't stop thinking about things. The feeling of not being needed, the feeling of being ignored. The soul is restless. Nervous. Afraid. I cannot concentrate at work and keep worrying about everything.",
        "Can muscle weakness be caused by anxiety? My arms and legs have been feeling really weak the past 2 days, and I'm getting worried sick that I have MS/ALS. Can't even carry out daily activities properly as I keep thinking that I'm gonna die. Help is much appreciated.",
        "Severe chronic stress. Can medication help? Hi I have had a series of extreme stress and my stress response is broken. My head hurts, I feel agitated, and can't think clearly, my muscles ache too, and I lack empathy. Is there anything that can break this out?",
        "Stress hives. How do I get rid of stress hives on my neck, shoulders and face? It isn't the kind where they last several days, but they keep coming back whenever things get busy at work.",
        "Anyone have bone or muscle pain that was stress or anxiety induced? I've been having joint and muscle pain that worries me constantly. Anyone else experienced this and found it was just the anxiety?",
    ],
    'Bipolar': [
        "CBT? DBT? Anyone had good experiences? I'm running out of drugs I can try. Vibryd, Lamictal, Latuda, and welbutrin worked for two years but then I crashed and i'm back to zero. Have tried many drugs but pdoc is useless for therapy otherwise. Does anything work on bipolar?",
        "Manic spending sprees. I'm diagnosed bipolar 2, and whenever I'm manic, I get reckless and spend money like mad. Does anyone have any advice to help with stopping this? It's killing me and makes my depressive episodes way worse.",
        "Manic episode? What are the signs? I feel like my over obsession over a small incident this morning is an indication of a manic episode because now I've been feeling sad and depleted. How long after diagnosis did you start being able to notice the signs?",
        'My amusing hypomania sign. Just a funny anecdote: one of my telltale signs of being hypomanic is midnight baking and cooking. Batches of cookies, muffins, whatever I can find a recipe for at two in the morning.',
        "New psychiatrist prescribed abilify, lexapro, and lamictal. Has anyone been on this combination, or any of them individually, for bipolar? I've only been on lamictal before and I'm not sure what to expect.",
    ],
    'Depression': [
        'I have been told that I should take anti-depressants, but they zombify me. I am already numb, what I need is to feel alive again. I am sick of waking up empty everyday. Nothing brings me joy anymore and I just go through the motions. How do you deal with your depression?',
        'Nothing feels worth doing anymore. I used to enjoy playing guitar and seeing friends, now I just lie in bed for hours scrolling. I get up, go to work, come home, and none of it means anything. I have felt this heavy for months now.',
        'I cannot remember the last time I actually felt happy about something. Everything is grey and flat. I keep telling people I am fine because explaining it is exhausting. I am so tired of pretending that things are okay when they are not.',
        'Woke up feeling completely empty again. I have no motivation to shower or eat properly. My friends invited me out and I made an excuse because being around people feels like too much effort. This has been going on for weeks.',
    ],
    'Normal': [
        'Had a pretty good week overall. Finished the project I was working on, went for a long walk on Saturday and met a few friends for coffee. Planning to start a new book tonight and maybe watch a film if there is time.',
        'Finally got around to reorganising the kitchen this weekend. Took much longer than expected but it looks great now. Also tried a new pasta recipe which turned out surprisingly well, so I will probably make that again next week.',
        'Just got back from a weekend trip to the coast with a few friends. The drive was long but the weather held up and we managed to get out on the water both days. Already talking about doing it again in the spring.',
        'Started a woodworking class on Tuesday evenings. First project is a small side table and I have already made a few mistakes, but the instructor is patient and the other people in the group are friendly.',
        'Been meaning to sort out the garden all summer and finally made a start on it today. Cleared the beds, planted a few things and repaired the fence panel that blew loose in the storm. Satisfying work.',
    ],
    'Suicidal': [
        'Going to end it tonight. I am tired of being in this endless loop, dying day after day. I know it is pretty pointless but the hurt is just too much to bear. Nobody would notice anyway. I have written the notes. It ends tonight.',
        'This is the perfect opportunity. Nobody is in the house. I want to die but I am too much of a coward to actually go through with it. I do not know what is stopping me anymore.',
        'I need someone to talk to. I am thinking about killing myself and there are no hotlines in my country. I need urgent help please help me.',
        'My heart is forever broken. I want to join them. I say my goodbyes in my head every night and wonder whether anyone would really notice if I was gone.',
    ],
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


# ── Input section — curated samples only, read-only ───────────────────────────
st.markdown('<div class="card-title">Select a sample post</div>', unsafe_allow_html=True)

col_cat, col_ex = st.columns([1, 2], gap="medium")

with col_cat:
    category = st.selectbox(
        "Source category",
        list(EXAMPLES.keys()),
        format_func=lambda c: f"{CLASS_ICONS.get(c, '')} {c}",
        key="sel_category",
    )

options = EXAMPLES[category]

with col_ex:
    choice = st.selectbox(
        "Example post",
        range(len(options)),
        format_func=lambda i: f"Example {i + 1} — {' '.join(options[i].split())[:70]}…",
        key="sel_example",
    )

text_input = options[choice]

# Read-only rendering: the sample text cannot be edited or replaced.
st.markdown(
    f'<div style="background:#F8FAFC;border:1px solid #E2E8F0;'
    f'border-left:4px solid {CLASS_COLOURS.get(category, "#6B7280")};'
    f'border-radius:8px;padding:14px 16px;margin:6px 0 4px 0;'
    f'font-size:0.95rem;line-height:1.65;color:#1F2937;">'
    f'{html.escape(text_input)}</div>',
    unsafe_allow_html=True,
)
st.caption(f"{len(text_input.split())} words · read-only sample — the model's "
           f"prediction below is computed independently of the category shown above.")

analyse = st.button("Analyse Text", type="primary")


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
    st.caption("Green bars support the prediction · red bars argue against it. "
               "New to these charts? See **💡 Understanding XAI** in the sidebar "
               "for a worked example.")
    tab_lime, tab_shap = st.tabs(["LIME Feature Attribution", "SHAP Feature Attribution"])

    with tab_lime:
        with st.spinner("Generating LIME explanation… (can take a minute on CPU-only machines)"):
            try:
                st.image(base64.b64decode(run_lime(text_input)['image']), use_container_width=True)
            except Exception as e:
                st.error(f"LIME explanation failed: {e}")

    with tab_shap:
        with st.spinner("Generating SHAP explanation… (can take a minute on CPU-only machines)"):
            try:
                st.image(base64.b64decode(run_shap(text_input)['image']), use_container_width=True)
            except Exception as e:
                st.error(f"SHAP explanation failed: {e}")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    ⚠️ <strong>Research prototype only.</strong>
    This tool is not a clinical diagnostic instrument and must not be used as a substitute
    for professional mental health assessment. Intended for academic research purposes only.<br>
    COM748 Masters Research Project · Fine-tuned BERT · LIME &amp; SHAP Explainability
</div>
""", unsafe_allow_html=True)
