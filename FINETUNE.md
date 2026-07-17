# Fine-Tuning BERT for Mental Health Detection — Step-by-Step Record

This document records exactly how the deployed model
([`code-world/bert-mental-health-detection`](https://huggingface.co/code-world/bert-mental-health-detection))
was produced, from raw data to the final checkpoint. The implementation lives in
`notebooks/06_train_bert.ipynb`; the data preparation steps live in
`notebooks/02_preprocess.ipynb`.

---

## 1. Data collection

Two public Kaggle datasets were downloaded via the Kaggle API (`notebooks/00_download_datasets.ipynb`):

| Dataset | Rows | Labels |
|---|---|---|
| [Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) | 53,043 | Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, Personality disorder |
| [Depression: Reddit Dataset (Cleaned)](https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned) | 7,731 | Depression / Non-Depression |

## 2. Merging and cleaning (`02_preprocess.ipynb`)

1. **Label unification** — the two label schemes were mapped onto 5 final classes:
   `Anxiety` + `Stress` → **Anxiety/Stress**; `Non-Depression` → **Normal**;
   `Personality disorder` was excluded. Final classes:
   **Anxiety/Stress · Bipolar · Depression · Normal · Suicidal**
2. **Merge** — both datasets concatenated: 60,774 rows.
3. **Deduplication** — exact duplicate texts removed → 50,178 rows.
4. **Short-text removal** — posts with too few tokens removed → 47,392 rows.
5. **Final corpus** after remaining preprocessing: **47,371 posts**
   (Depression 15,013 · Normal 13,450 · Suicidal 10,606 · Anxiety/Stress 5,823 · Bipolar 2,500).
6. **Label encoding** — class names encoded to integers 0–4 with a scikit-learn
   `LabelEncoder`, persisted to `datasets/processed/splits/label_encoder.pkl`.

## 3. Train / validation / test split

Split **once**, at preprocessing time, and reused identically by every model in the study
(classical ML, deep learning and BERT) so all results are comparable:

```python
RANDOM_SEED = 42

# 70% train | 15% validation | 15% test — stratified by class
train, temp = train_test_split(df,   test_size=0.30, random_state=RANDOM_SEED,
                               stratify=df['label_encoded'])
val,  test  = train_test_split(temp, test_size=0.50, random_state=RANDOM_SEED,
                               stratify=temp['label_encoded'])
```

| Split | Posts | Share |
|---|---|---|
| Train | 33,159 | 70% |
| Validation | 7,106 | 15% |
| Test | 7,106 | 15% |

- **Stratification** preserves the class ratios in every split, which matters because the
  corpus is imbalanced (Depression has ~6× more posts than Bipolar).
- The **test split was touched exactly once** — for the final evaluation of the selected
  checkpoint. All tuning decisions used the validation split only.
- Saved to `datasets/processed/splits/{train,val,test}.csv`.

## 4. Input text choice — raw, not cleaned

The classical ML models use a heavily cleaned `clean_text` column (lower-cased, URLs and
punctuation stripped, stopwords removed, lemmatised). **BERT was fine-tuned on the raw
`text` column instead**, because:

- BERT ships with its own WordPiece tokenizer trained on natural sentences;
- stopword removal deletes negation and function words ("not", "can't", "I") that carry
  the syntactic signal transformers exploit;
- lemmatisation destroys tense/aspect cues (e.g. "dying" → "die").

This choice was validated empirically: the earlier frozen-BERT experiment on cleaned text
reached macro-F1 0.680, while fine-tuning on raw text reached 0.841.

## 5. Tokenisation

- Tokenizer: `bert-base-uncased` WordPiece
- `max_length = 128` tokens (EDA showed this covers the bulk of posts), with truncation
  and padding to fixed length
- All three splits tokenised once up-front for training throughput

## 6. Model

- Base: **`bert-base-uncased`** (12 layers, 768 hidden, 110M parameters)
- Head: `AutoModelForSequenceClassification` with `num_labels = 5`
  (linear classification layer on the pooled output, newly initialised)
- **All layers trainable** — full end-to-end fine-tuning, no freezing

## 7. Class-weighted loss

The training set is imbalanced (Depression 10,509 vs Bipolar 1,750 posts). Cross-entropy
was weighted with scikit-learn's `compute_class_weight('balanced', ...)` — each class's
weight is inversely proportional to its frequency — so minority classes (Bipolar,
Anxiety/Stress) contribute proportionally to the gradient.

## 8. Optimisation

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| LR schedule | Linear warmup over first 10% of steps, then linear decay to 0 |
| Batch size (training) | 16 |
| Batch size (evaluation) | 64 |
| Epochs | 3 |
| Gradient clipping | max-norm 1.0 |
| Seed | 42 (torch + numpy) |
| Hardware | Apple M2 Pro GPU (PyTorch MPS backend) |

This follows the original BERT fine-tuning recipe (Devlin et al., 2019).

## 9. Checkpoint selection

After **every epoch** the model was evaluated on the validation split. The checkpoint with
the **best validation macro-F1** was saved (`model.save_pretrained`) to
`models/saved/bert_finetuned/` and the final test evaluation used that checkpoint —
not simply the last epoch.

Per-epoch record (`results/bert_finetune_history.csv`):

| Epoch | Train loss | Val accuracy | Val macro-F1 |
|---|---|---|---|
| 1 | 0.691 | 0.812 | 0.808 |
| 2 | 0.357 | 0.828 | 0.830 |
| 3 | 0.236 | 0.828 | 0.833 ← selected |

Validation F1 plateaus by epoch 3 with no sign of overfitting, so the 3-epoch budget was
sufficient.

## 10. Final test-set evaluation (single pass)

| Metric | Score |
|---|---|
| Accuracy | **83.4%** |
| Macro F1 | **0.841** |

Per-class F1: Normal **0.953** · Anxiety/Stress **0.882** · Bipolar **0.858** ·
Depression **0.770** · Suicidal **0.742**
(full report: `results/classification_report_bert.csv`;
confusion matrix: `results/charts/confusion_bert.png`)

### Comparison against all baselines (same test split)

| Model | Accuracy | Macro F1 |
|---|---|---|
| **BERT (fine-tuned)** | **83.4%** | **0.841** |
| XGBoost (TF-IDF) | 77.2% | 0.775 |
| Logistic Regression (TF-IDF) | 76.7% | 0.773 |
| CNN | 73.3% | 0.727 |
| BERT (frozen encoder + LR) | 69.9% | 0.680 |
| BiLSTM | 64.1% | 0.629 |

Fine-tuning added **+16.1 macro-F1 points** over the frozen encoder and **+6.6 points**
over the strongest classical baseline.

## 11. Artefacts produced

| Artefact | Path |
|---|---|
| Fine-tuned model + tokenizer | `models/saved/bert_finetuned/` |
| Hugging Face mirror (public) | `code-world/bert-mental-health-detection` |
| Per-epoch history | `results/bert_finetune_history.csv` |
| Test classification report | `results/classification_report_bert.csv` |
| Results JSON (frozen + fine-tuned) | `models/saved/bert_results.json` |
| Confusion matrix chart | `results/charts/confusion_bert.png` |
