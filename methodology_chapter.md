# Chapter 3: Research Methodology

## 3.1 Introduction

This chapter describes the methodology adopted to design, develop, and evaluate an
explainable transformer-based framework for the early detection of mental health
disorders from social media text. It sets out the overall research design, the
computational environment in which the work was conducted, the datasets that were
collected, and the exploratory data analysis (EDA) carried out to understand the
characteristics of those datasets prior to modelling. The decisions taken during
data collection and exploration directly inform the preprocessing, feature
engineering, and modelling stages reported in the subsequent chapters.

## 3.2 Research Design

The research follows an **experimental, quantitative methodology** organised as a
sequential, reproducible pipeline. The work is structured around the
widely-used stages of a data science / machine learning project — data collection,
data understanding (EDA), data preprocessing, feature engineering, model
development, model evaluation, and deployment — which mirror the CRISP-DM process
model and the objectives set out in the project proposal.

Each stage was implemented as a self-contained Jupyter notebook so that the pipeline
could be executed in a fixed order, intermediate artefacts (cleaned data, feature
matrices, trained models) could be persisted to disk, and individual stages could be
re-run independently without repeating the whole pipeline. The high-level workflow
is summarised below:

| Stage | Notebook | Purpose |
|---|---|---|
| Data collection | `00_download_datasets` | Programmatic download of the two Kaggle datasets |
| Data understanding (EDA) | `01_explore_datasets` | Class distribution, text-length, and sample inspection on the raw data |
| Data preprocessing | `02_preprocess` | Cleaning, label harmonisation, deduplication, and train/validation/test splitting |
| Feature engineering | `03_features` | TF-IDF representations for the classical models |
| Model development | `04`–`06` | Logistic Regression, XGBoost, CNN, BiLSTM, and fine-tuned BERT |
| Evaluation & comparison | `07_compare` | Comparative analysis across all models |
| Explainability | `08_explainability` | SHAP and LIME interpretation of predictions |

The comparative element of the design is central: classical machine learning
(Logistic Regression, XGBoost) and deep learning (CNN, BiLSTM) models act as
baselines against which the transformer-based model (BERT) is benchmarked using a
common set of evaluation metrics, ensuring a fair, like-for-like comparison.

## 3.3 Development Environment and Tools

The implementation was carried out in **Python** within a dedicated virtual
environment to guarantee reproducibility and dependency isolation. The principal
libraries used across the pipeline were:

- **Data handling and analysis:** pandas, NumPy
- **Visualisation:** Matplotlib, seaborn
- **Natural language processing:** NLTK (tokenisation, stop-word removal,
  lemmatisation), spaCy
- **Classical machine learning:** scikit-learn, XGBoost, imbalanced-learn
- **Deep learning and transformers:** PyTorch, Hugging Face Transformers
- **Explainable AI:** SHAP, LIME
- **Deployment:** FastAPI (backend) and Streamlit (frontend)
- **Data acquisition:** the Kaggle API client

A fixed random seed (`42`) was applied throughout to ensure that data splitting and
model training are reproducible.

## 3.4 Data Collection

In line with the project's ethical commitments, only **publicly available and
anonymised** datasets were used; no data was scraped directly from social media
platforms and no attempt was made to identify individual users. Two complementary
datasets were obtained from the Kaggle repository, downloaded programmatically using
the Kaggle API so that the acquisition step is automated and repeatable.

### 3.4.1 Dataset 1 — Sentiment Analysis for Mental Health

The primary dataset is the *Sentiment Analysis for Mental Health* dataset
(Sarkar, 2022) [8]. It consists of **53,043** labelled text statements drawn from
social media content and contains three columns (an index column, the text
`statement`, and the `status` label). The dataset spans **seven** mental-health
categories: *Normal, Depression, Suicidal, Anxiety, Bipolar, Stress,* and
*Personality disorder*. This dataset provides the multi-class signal that is core to
the project's aim of detecting multiple conditions rather than a single binary
outcome.

### 3.4.2 Dataset 2 — Depression Reddit Cleaned

The second dataset is the *Depression Reddit Cleaned* dataset
(InfamousCoder, 2023) [7], containing **7,731** cleaned Reddit posts with two
columns: the post text (`clean_text`) and a binary label (`is_depression`). It is a
balanced binary dataset (Depression vs. Non-Depression) and was incorporated to
strengthen the representation of depression-related and non-clinical ("Normal")
language, complementing the broader multi-class dataset.

Combining the two sources yields a corpus of **60,774** raw samples prior to
cleaning, providing both breadth of conditions and additional examples of the most
prevalent classes.

### 3.4.3 Ethical Considerations

Consistent with the ACM Code of Ethics and the BCS Code of Conduct, the datasets are
publicly licensed and already anonymised. All data is used exclusively for academic
research, and the resulting system is positioned as a **decision-support tool**, not
a clinical diagnostic instrument, in recognition of the sensitivity of mental-health
data and the consequences of misclassification.

## 3.5 Exploratory Data Analysis

Before any cleaning or modelling, an exploratory analysis was performed on the **raw**
datasets to characterise their size, class balance, and textual properties, and to
surface issues that the preprocessing stage would need to address. Three lenses were
applied to each dataset: **class distribution**, **text-length distribution**, and
**qualitative inspection of sample texts**.

### 3.5.1 Dataset 1 — Class Distribution

Dataset 1 contains no missing values but is **markedly class-imbalanced**. The
distribution across the seven raw classes is shown below.

| Class | Count |
|---|---|
| Normal | 16,351 |
| Depression | 15,404 |
| Suicidal | 10,653 |
| Anxiety | 3,888 |
| Bipolar | 2,877 |
| Stress | 2,669 |
| Personality disorder | 1,201 |

The three largest classes (Normal, Depression, Suicidal) account for the large
majority of samples, while *Personality disorder* is severely under-represented with
only 1,201 examples — roughly **13 times** smaller than the largest class. This
imbalance is a key finding, as it risks biasing classifiers toward the majority
classes and motivates the class-harmonisation and balancing decisions taken later
(e.g. removing the sparsest class and merging closely related categories).

### 3.5.2 Dataset 1 — Text-Length Distribution

Text length was measured as the number of whitespace-delimited words per statement.
The distribution is **heavily right-skewed**:

| Statistic | Words |
|---|---|
| Mean | 112.4 |
| Std. dev. | 163.4 |
| Minimum | 0 |
| 25th percentile | 15 |
| Median | 61 |
| 75th percentile | 147.5 |
| Maximum | 6,300 |

Although the median statement is around 61 words, a long tail of very long posts
(up to 6,300 words) pulls the mean well above the median. The presence of a minimum
of 0 words also indicates the existence of empty or whitespace-only entries. These
observations directly justify two later design choices: removing empty/very short
texts during cleaning, and imposing a maximum sequence length when tokenising input
for the deep-learning and transformer models.

### 3.5.3 Dataset 1 — Sample Inspection

A qualitative review of representative samples from each class confirmed that the
texts are authentic, free-form social-media language containing informal spelling,
trigger warnings, emotional expression, and occasional formatting artefacts (e.g.
Markdown, embedded links). This reinforced the need for a robust text-cleaning
pipeline (lower-casing, URL and punctuation removal, lemmatisation, and stop-word
removal).

### 3.5.4 Dataset 2 — Class Distribution and Text Length

Dataset 2 is, by contrast, **well balanced** between its two classes and contains no
missing values:

| Class | Count |
|---|---|
| Non-Depression | 3,900 |
| Depression | 3,831 |

Its texts are shorter on average than those in Dataset 1, with a similarly
right-skewed distribution:

| Statistic | Words |
|---|---|
| Mean | 74.6 |
| Std. dev. | 144.4 |
| Minimum | 1 |
| 25th percentile | 12 |
| Median | 22 |
| 75th percentile | 76 |
| Maximum | 4,239 |

The shorter median length (22 words) reflects the cleaned, post-level nature of this
Reddit dataset and the fact that it has already undergone some preprocessing by its
original authors.

### 3.5.5 Combined Overview

Taken together, the two datasets are summarised below.

| Dataset | Rows | Classes | Avg. words | Missing |
|---|---|---|---|---|
| Sentiment Analysis for Mental Health | 53,043 | 7 | 112.4 | 0 |
| Depression Reddit Cleaned | 7,731 | 2 | 74.6 | 0 |
| **Combined** | **60,774** | — | — | 0 |

### 3.5.6 Key Findings and Implications for Preprocessing

The exploratory analysis produced several findings that shaped the remainder of the
methodology:

1. **Class imbalance** in Dataset 1 — particularly the very small *Personality
   disorder* class — must be addressed to avoid biased classifiers.
2. **Overlapping / closely related categories** (e.g. *Anxiety* and *Stress*) suggest
   that consolidating semantically similar labels could yield more robust classes.
3. **Skewed text lengths**, including empty and extremely long entries, require
   filtering of empty/very short texts and a capped maximum sequence length for the
   neural models.
4. **Differing label schemes** across the two datasets (multi-class vs. binary) must
   be reconciled into a single, consistent label set before they can be merged.
5. **Absence of missing values** in both raw datasets means no imputation is required.

These findings are carried directly into the data-preprocessing stage, where the
labels are harmonised, the datasets merged, noisy and duplicate records removed, and
the corpus split into stratified training, validation, and test sets.
