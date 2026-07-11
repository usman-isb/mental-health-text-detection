"""Results & EDA page — every table and chart produced by the pipeline."""

import pandas as pd
import streamlit as st

import shared
from shared import CHARTS_DIR, RESULTS_DIR, section, stat_row

shared.inject_css()

shared.header(
    "Results &amp; Exploratory Data Analysis",
    "Complete experimental results of the study — dataset exploration, model comparison, "
    "fine-tuning behaviour, per-class evaluation and explainability analysis.",
    "6 Models · 47,371 Posts · SHAP &amp; LIME",
)


@st.cache_data
def load_csv(path, **kw):
    return pd.read_csv(path, **kw)


def img(name, caption=None):
    p = CHARTS_DIR / name
    if p.exists():
        st.image(str(p), use_container_width=True, caption=caption)
    else:
        st.info(f"Chart not found: {name} — run the corresponding notebook.")


# ══════════════════════════════ 1 · Datasets & EDA ═══════════════════════════
section("1 · Datasets and exploratory data analysis",
        "Two public Kaggle datasets were merged, deduplicated and mapped onto five classes. "
        "EDA examined class balance and text length before any modelling decisions were made.")

stat_row([
    ("53,043", "Dataset 1 posts"),
    ("7,731", "Dataset 2 posts"),
    ("47,371", "After cleaning"),
    ("5", "Classes"),
    ("70 / 15 / 15", "Train / Val / Test %"),
])

st.markdown("""
| Step | Rows | Detail |
|---|---|---|
| Dataset 1 — *Sentiment Analysis for Mental Health* | 53,043 | 7 raw labels (Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, Personality disorder) |
| Dataset 2 — *Depression: Reddit (Cleaned)* | 7,731 | Binary (Depression / Non-Depression) |
| Merged | 60,774 | Labels unified — Anxiety + Stress → Anxiety/Stress |
| Duplicates removed | 50,178 | Exact-text duplicates dropped |
| Short texts removed | 47,392 | Posts with too few tokens excluded |
| **Final corpus** | **47,371** | Depression 15,013 · Normal 13,450 · Suicidal 10,606 · Anxiety/Stress 5,823 · Bipolar 2,500 |
""")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Dataset 1 — Mental Health Sentiment", "Dataset 2 — Reddit Depression"])
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        img("eda_ds1_class_distribution.png", "Class distribution — the 7 raw labels are heavily imbalanced.")
    with c2:
        img("eda_ds1_text_length.png", "Text length distribution — long tail; 128 tokens covers the bulk of posts.")
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        img("eda_ds2_class_distribution.png", "Class distribution — near-perfectly balanced binary dataset.")
    with c2:
        img("eda_ds2_text_length.png", "Text length distribution.")

st.markdown(
    '<div class="section-sub">Key EDA findings: (1) strong class imbalance — Depression has ~6× more posts '
    'than Bipolar, motivating class-weighted training and macro-F1 as the headline metric; '
    '(2) most posts fit comfortably within a 128-token window, which was adopted for all sequence models.</div>',
    unsafe_allow_html=True)


# ══════════════════════════════ 2 · Model comparison ═════════════════════════
section("2 · Model comparison",
        "Six models spanning three families — classical ML (TF-IDF features), deep learning "
        "(trained embeddings) and transformers — evaluated on the same held-out test split "
        "(n = 7,106). Macro-F1 is the primary metric due to class imbalance.")

comp = load_csv(RESULTS_DIR / "model_comparison.csv")
best = comp.sort_values("macro_f1", ascending=False).iloc[0]

stat_row([
    (best["model"], "Best model", True),
    (f"{best['accuracy']:.1%}", "Test accuracy", True),
    (f"{best['macro_f1']:.3f}", "Macro F1", True),
    ("+16.1 pts", "F1 gain from fine-tuning"),
])

show = comp.sort_values("macro_f1", ascending=False).reset_index(drop=True)
show.index = show.index + 1
show.columns = ["Model", "Accuracy", "Macro F1"]
st.dataframe(
    show.style.format({"Accuracy": "{:.4f}", "Macro F1": "{:.4f}"})
        .background_gradient(subset=["Macro F1"], cmap="Blues"),
    use_container_width=True,
)

c1, c2 = st.columns([3, 2])
with c1:
    img("model_comparison_bar.png", "Accuracy and macro-F1 for all six models.")
with c2:
    img("model_comparison_radar.png", "Radar view of the same comparison.")

st.markdown(
    '<div class="section-sub">The fine-tuned BERT leads on both metrics. Notably, classical '
    'TF-IDF models (XGBoost, Logistic Regression) beat both deep learning baselines and the '
    '<em>frozen</em> BERT — transformer advantage only materialises once the encoder is '
    'fine-tuned on the task.</div>', unsafe_allow_html=True)


# ══════════════════════════════ 3 · BERT fine-tuning ═════════════════════════
section("3 · BERT fine-tuning behaviour",
        "bert-base-uncased fine-tuned end-to-end (3 epochs, AdamW lr 2e-5, linear warmup, "
        "class-weighted loss, batch 16, max 128 tokens). Best checkpoint chosen by validation macro-F1.")

hist_path = RESULTS_DIR / "bert_finetune_history.csv"
if hist_path.exists():
    hist = load_csv(hist_path)
    c1, c2 = st.columns([2, 3])
    with c1:
        h = hist.copy()
        h.columns = ["Epoch", "Train loss", "Val accuracy", "Val macro F1"]
        st.dataframe(h.style.format({"Train loss": "{:.4f}", "Val accuracy": "{:.4f}",
                                     "Val macro F1": "{:.4f}"}),
                     use_container_width=True, hide_index=True)
        st.markdown(
            '<div class="section-sub">Training loss falls steadily while validation F1 '
            'plateaus by epoch 3 — no overfitting within the budget.</div>',
            unsafe_allow_html=True)
    with c2:
        chart_df = hist.set_index("epoch")[["val_acc", "val_macro_f1"]]
        chart_df.columns = ["Validation accuracy", "Validation macro F1"]
        st.line_chart(chart_df, height=260)

st.markdown("""
| Configuration | Test accuracy | Test macro F1 |
|---|---|---|
| BERT frozen ([CLS] embeddings + Logistic Regression, cleaned text) | 69.9% | 0.680 |
| **BERT fine-tuned (end-to-end, raw text, class-weighted loss)** | **83.4%** | **0.841** |
""")


# ══════════════════════════════ 4 · Per-class evaluation ═════════════════════
section("4 · Per-class evaluation",
        "Classification reports and confusion matrices on the test split.")

rep_tabs = st.tabs(["BERT (fine-tuned)", "XGBoost", "Logistic Regression"])
for tab, fname in zip(rep_tabs, ["classification_report_bert.csv",
                                 "classification_report_xgboost.csv",
                                 "classification_report_logistic_regression.csv"]):
    with tab:
        p = RESULTS_DIR / fname
        if p.exists():
            rep = load_csv(p, index_col=0)
            st.dataframe(rep.style.format("{:.3f}", subset=["precision", "recall", "f1-score"])
                            .background_gradient(subset=["f1-score"], cmap="Blues"),
                         use_container_width=True)
        else:
            st.info(f"Report not found: {fname}")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="card-title">Confusion matrices (row-normalised)</div>', unsafe_allow_html=True)

cm_tabs = st.tabs(["BERT (fine-tuned)", "XGBoost", "Logistic Regression", "CNN", "BiLSTM"])
for tab, fname in zip(cm_tabs, ["confusion_bert.png", "confusion_xgboost.png",
                                "confusion_logistic_regression.png",
                                "confusion_cnn.png", "confusion_bilstm.png"]):
    with tab:
        c, _ = st.columns([3, 1])
        with c:
            img(fname)

st.markdown(
    '<div class="section-sub">Across all models the dominant error is the '
    'Depression ↔ Suicidal confusion — clinically the most overlapping pair. The fine-tuned '
    'BERT achieves the cleanest diagonal, with Normal (F1 0.953) and Anxiety/Stress (0.882) '
    'almost fully separated.</div>', unsafe_allow_html=True)

st.markdown('<div class="card-title">Deep learning training curves</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    img("training_curve_cnn.png", "CNN — loss and validation metrics per epoch.")
with c2:
    img("training_curve_bilstm.png", "BiLSTM — loss and validation metrics per epoch.")


# ══════════════════════════════ 5 · Explainability ═══════════════════════════
section("5 · Explainability — SHAP and LIME on the fine-tuned BERT",
        "Post-hoc explanations generated for the deployed model itself: SHAP Partition explainer "
        "with a text masker (token-level) and LIME (word-level perturbation).")

shared.explainer_expander()

img("shap_global_importance.png",
    "Global token importance — mean |SHAP| per token per class, aggregated over a stratified "
    "sample of 100 test posts.")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
xt1, xt2, xt3 = st.tabs(["SHAP — one sample per class", "LIME — one sample per class",
                         "SHAP vs LIME side by side"])
with xt1:
    img("shap_individual_per_class.png")
with xt2:
    img("lime_individual_per_class.png")
with xt3:
    c, _ = st.columns([4, 1])
    with c:
        img("shap_vs_lime_comparison.png")

agree_path = RESULTS_DIR / "shap_lime_agreement.csv"
if agree_path.exists():
    st.markdown('<div class="card-title">SHAP–LIME agreement (top-12 tokens per method)</div>',
                unsafe_allow_html=True)
    agree = load_csv(agree_path)
    agree.columns = ["Class", "Shared tokens", "Shared count", "Jaccard"]
    st.dataframe(agree.style.format({"Jaccard": "{:.3f}"})
                    .background_gradient(subset=["Jaccard"], cmap="Blues"),
                 use_container_width=True, hide_index=True)
    st.markdown(
        f'<div class="section-sub">Mean Jaccard overlap: <strong>{agree["Jaccard"].mean():.3f}</strong> '
        '— moderate agreement. Both methods consistently surface the clinically salient terms '
        '(e.g. “anxiety”, “killing”, “hopeless”); they diverge in the tail because SHAP attributes '
        'over sub-word token spans while LIME weights whole words. Agreement is weakest for '
        'Bipolar, the smallest class.</div>', unsafe_allow_html=True)

st.markdown(shared.FOOTER, unsafe_allow_html=True)
