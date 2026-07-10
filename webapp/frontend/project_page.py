"""Project Overview page — aim, objectives, methodology, architecture, ethics."""

import streamlit as st

import shared
from shared import section, stat_row

shared.inject_css()

shared.header(
    "Project Overview",
    "Explainable Transformer Based Framework for Early Detection of Mental Health "
    "Disorders from Social Media Text.",
    "COM748 Masters Research Project",
)

# ══════════════════════════════ Aim ══════════════════════════════════════════
section("Aim")
st.markdown("""
To design, develop and evaluate an **explainable, transformer-based framework** that detects
signs of mental health disorders in social media text — comparing classical machine learning,
deep learning and transformer approaches, and deploying the best model in an interactive web
application whose predictions are transparent through **SHAP** and **LIME** explanations.
""")

stat_row([
    ("5", "Mental health classes"),
    ("6", "Models compared"),
    ("47,371", "Posts analysed"),
    ("83.4%", "Best accuracy", True),
    ("0.841", "Best macro F1", True),
])

# ══════════════════════════════ Objectives ═══════════════════════════════════
section("Objectives")
st.markdown("""
1. **Collect and curate** publicly available social media datasets covering five mental health
   states: Anxiety/Stress, Bipolar, Depression, Normal and Suicidal.
2. **Explore and preprocess** the data — deduplication, label unification, cleaning and
   stratified train/validation/test splitting.
3. **Develop and compare** three model families under a common evaluation protocol:
   classical ML (Logistic Regression, XGBoost on TF-IDF), deep learning (CNN, BiLSTM) and
   transformers (BERT — frozen and fine-tuned).
4. **Explain** the best model's predictions with two complementary post-hoc methods
   (SHAP and LIME) and quantify their agreement.
5. **Deploy** the best model in a web application providing real-time, explained predictions.
""")

# ══════════════════════════════ Methodology ══════════════════════════════════
section("Methodology",
        "An experimental, quantitative pipeline mirroring the CRISP-DM process model. Each stage "
        "is a self-contained notebook that persists its artefacts, so stages re-run independently.")

st.markdown("""
| Stage | Notebook | Purpose |
|---|---|---|
| Data collection | `00_download_datasets` | Download both Kaggle datasets via the Kaggle API |
| Data understanding | `01_explore_datasets` | EDA — class balance, text length, sample inspection |
| Preprocessing | `02_preprocess` | Merge, dedupe, unify labels, clean, stratified 70/15/15 split |
| Feature engineering | `03_features` | TF-IDF matrices for the classical ML models |
| Classical ML | `04_train_ml` | Logistic Regression and XGBoost |
| Deep learning | `05_train_dl` | CNN and BiLSTM with trained embeddings |
| Transformer | `06_train_bert` | BERT fine-tuned end-to-end (+ frozen baseline) |
| Evaluation | `07_compare` | Unified comparison on the held-out test set |
| Explainability | `08_explainability` | SHAP + LIME analysis of the best model |
""")

# ══════════════════════════════ Datasets ═════════════════════════════════════
section("Datasets")
st.markdown("""
| | Dataset 1 | Dataset 2 |
|---|---|---|
| **Name** | [Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) | [Depression: Reddit Dataset (Cleaned)](https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned) |
| **Source** | Kaggle (aggregated social media statements) | Kaggle (Reddit posts) |
| **Size** | 53,043 posts | 7,731 posts |
| **Labels** | 7 raw statuses | Binary (depression / non-depression) |

After merging, label unification (Anxiety + Stress → Anxiety/Stress), deduplication and
removal of very short posts, the final corpus contains **47,371 posts** across five classes,
split 70/15/15 into train (33,159), validation (7,106) and test (7,106) sets with stratification.
""")

# ══════════════════════════════ Best model ═══════════════════════════════════
section("Final model",
        "bert-base-uncased (110M parameters) fine-tuned end-to-end for 3 epochs with AdamW "
        "(lr 2e-5), linear warmup/decay, class-weighted cross-entropy and 128-token inputs. "
        "Selected by validation macro-F1 and evaluated once on the held-out test set.")

st.markdown("""
- **Test accuracy 83.4% · macro F1 0.841** — best of all six models compared
- Fine-tuning added **+16.1 F1 points** over the same encoder used frozen, and **+6.6 F1 points**
  over the strongest classical baseline (XGBoost)
- Trained on **raw text** (not the lemmatised/stopword-stripped version used for TF-IDF models),
  preserving the negation and function-word cues transformers rely on
""")

# ══════════════════════════════ Architecture ═════════════════════════════════
section("System architecture")
st.markdown("""
```
Kaggle datasets ──▶ Notebook pipeline (00–08) ──▶ Persisted artefacts
                                                    ├── models/saved/bert_finetuned/   (fine-tuned BERT)
                                                    ├── datasets/processed/            (splits, encoders)
                                                    └── results/                       (metrics, charts)
                                                                    │
                    ┌───────────────────────────────────────────────┘
                    ▼
      FastAPI backend (port 8000)                        Streamlit frontend (port 8501)
      ├── POST /predict        ◀────────────────────────  🧠 Analyser (this app)
      ├── POST /explain/lime   ◀────────────────────────  📊 Results & EDA
      └── POST /explain/shap   ◀────────────────────────  📋 Project Overview
```
""")
st.markdown("""
- **Backend** — FastAPI serving the fine-tuned BERT on Apple-Silicon GPU (MPS), with LIME and
  SHAP (Partition explainer + text masker) computed on demand per request.
- **Frontend** — Streamlit multi-page app: live analyser with input validation, full results
  dashboard, and this overview page.
""")

# ══════════════════════════════ Tech stack ═══════════════════════════════════
section("Technology stack")
st.markdown("""
| Layer | Tools |
|---|---|
| Language | Python 3.13 |
| Data | pandas · NumPy · Kaggle API |
| Classical ML | scikit-learn (TF-IDF, Logistic Regression) · XGBoost |
| Deep learning | PyTorch (CNN, BiLSTM) |
| Transformer | Hugging Face Transformers (`bert-base-uncased`) · MPS GPU acceleration |
| Explainability | SHAP (Partition/Text) · LIME |
| Deployment | FastAPI · Uvicorn · Streamlit |
| Visualisation | Matplotlib · Seaborn |
""")

# ══════════════════════════════ Ethics ═══════════════════════════════════════
section("Ethical considerations and limitations")
st.markdown("""
- **Not a diagnostic tool.** Predictions reflect linguistic patterns in self-reported social
  media text, not clinical assessments. The app displays a persistent disclaimer.
- **Data provenance.** Both datasets are public and anonymised; no new user data is collected
  and analysed text is not stored.
- **Class imbalance.** Bipolar is under-represented (2,500 posts) and shows the weakest
  explainability agreement — findings for this class carry more uncertainty.
- **Domain shift.** Performance on other platforms, languages or demographics is untested.
- **Known error mode.** Depression ↔ Suicidal confusion is the dominant residual error;
  in a real triage setting the cost of that specific error is asymmetric and would require
  a recall-prioritised threshold for the Suicidal class.
""")

# ══════════════════════════════ Future work ══════════════════════════════════
section("Future work")
st.markdown("""
- Domain-specific encoders (e.g. MentalBERT / RoBERTa) and larger context windows
- Multi-label formulation to model co-occurring conditions
- Temporal modelling of user history for genuinely *early* detection rather than single-post classification
- Human-grounded evaluation of the SHAP/LIME explanations with clinicians
""")

st.markdown(shared.FOOTER, unsafe_allow_html=True)
