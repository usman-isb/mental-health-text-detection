"""
Streamlit page — how to read the LIME and SHAP explanations.
Explains both methods once, with one worked example, so the charts on the
Analyser page need no per-chart commentary.
"""

import streamlit as st

import shared
from shared import CHARTS_DIR, section

shared.inject_css()

EXAMPLE = ("It's been 3 weeks I feel really tired. The heart is pounding, the body "
           "is shaky. In this brain it doesn't stop thinking about things. The "
           "feeling of not being needed, the feeling of being ignored. The soul is "
           "restless. Nervous. Afraid. I cannot concentrate at work and keep "
           "worrying about everything.")

shared.header(
    "Understanding the Explanations",
    "How to read the LIME and SHAP charts — one worked example.",
    "Explainable AI · LIME &amp; SHAP",
)

# ── Worked example ────────────────────────────────────────────────────────────
section("The example", "One post, explained by both methods.")
st.info(EXAMPLE)

col1, col2 = st.columns([1, 1], gap="large")
with col1:
    st.markdown(
        '<div class="result-card" style="background:#D97706">'
        '<div class="result-label">Model prediction</div>'
        '<div class="result-class">😰 Anxiety/Stress</div>'
        '<div class="result-conf">Confidence &nbsp;<strong>99.9%</strong></div>'
        '</div>', unsafe_allow_html=True)
with col2:
    st.markdown("The words *anxiety* and *stress* never appear in this text, so the "
                "model could not have matched the label directly — it read the "
                "described symptoms.")

# ── How to read the charts ────────────────────────────────────────────────────
section("Reading any chart", "Three rules cover both methods.")
st.markdown("""
- **Each bar is one word** from the submitted text.
- **Green supports the prediction, red argues against it.**
- **Bar length is influence, not certainty** — order matters, the axis value does not.
""")

# ── LIME ──────────────────────────────────────────────────────────────────────
section("LIME", "What if these words were missing?")
st.markdown("LIME removes words, re-runs the model, and measures what changed. It "
            "works on whole words, so every bar is a word you can find in your text.")
st.image(str(CHARTS_DIR / 'xai_example_lime.png'), use_container_width=True)
st.caption("Strongest evidence: *restless*, *worrying*, *concentrate*, *heart*, "
           "*Nervous*, *Afraid* — the symptoms a human reader would also pick.")

# ── SHAP ──────────────────────────────────────────────────────────────────────
section("SHAP", "How is the prediction shared among the words?")
st.markdown("SHAP fairly distributes the predicted probability across the text using "
            "Shapley values. It works on the token spans BERT actually reads, so word "
            "fragments (*rest*, *##less*) can appear.")
st.image(str(CHARTS_DIR / 'xai_example_shap.png'), use_container_width=True)
st.caption("The same text and prediction, explained independently of LIME.")

# ── Why both ──────────────────────────────────────────────────────────────────
section("Why both are shown", "Agreement is evidence; disagreement is a caution.")
st.markdown("""
| Class | Top-word overlap (Jaccard) |
|---|---|
| Anxiety/Stress | 0.333 |
| Depression | 0.333 |
| Suicidal | 0.235 |
| Normal | 0.211 |
| Bipolar | 0.111 |
""")
st.markdown("Where both methods highlight the same words, the evidence is strong. "
            "Overlap is highest for classes with a distinctive vocabulary and lowest "
            "for Bipolar, whose posts often discuss medication rather than emotion.")

# ── Limits ────────────────────────────────────────────────────────────────────
section("What they do not tell you", "Limits worth stating.")
st.markdown("""
- Explanations show what the model reacted to — **not** clinical reasoning.
- A clear explanation does **not** mean the prediction is correct; a wrong
  prediction still produces a confident-looking chart.
- Both methods sample, so lower-ranked words shift slightly between runs.
- Each explanation needs hundreds of model passes, so it is generated on request.
""")

st.markdown("""
<div class="app-footer">
    ⚠️ <strong>Research prototype only.</strong>
    Explanations support transparency and academic evaluation — not diagnosis.<br>
    COM748 Masters Research Project · Fine-tuned BERT · LIME &amp; SHAP Explainability
</div>
""", unsafe_allow_html=True)
