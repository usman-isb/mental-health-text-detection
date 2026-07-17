"""
FastAPI Backend — Mental Health Detection
Model: BERT fine-tuned (best performing — F1 0.841, Accuracy 83.41%)
Classes: Anxiety/Stress, Bipolar, Depression, Normal, Suicidal
Explainability: SHAP Partition explainer (Text masker) + LIME
Endpoints:
  GET  /              → health check
  POST /predict       → prediction + probabilities
  POST /explain/lime  → LIME explanation as base64 image
  POST /explain/shap  → SHAP explanation as base64 image
"""

import os, io, base64, pickle, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = os.path.join(os.path.dirname(__file__), '..', '..')
SPLITS_DIR = os.path.join(BASE, 'datasets', 'processed', 'splits')
MODELS_DIR = os.path.join(BASE, 'models', 'saved')
FT_DIR     = os.path.join(MODELS_DIR, 'bert_finetuned')

# ── Load artifacts ────────────────────────────────────────────────────────────
print("Loading model artifacts ...")

with open(os.path.join(SPLITS_DIR, 'label_encoder.pkl'), 'rb') as f:
    le = pickle.load(f)
CLASS_NAMES = list(le.classes_)
NUM_CLASSES = len(CLASS_NAMES)

DEVICE = torch.device('mps' if torch.backends.mps.is_available()
                      else 'cuda' if torch.cuda.is_available() else 'cpu')

# On CPU-only machines (e.g. Intel Macs) each explanation needs hundreds of BERT
# forward passes — use smaller sampling budgets there to keep response times sane.
IS_CPU           = DEVICE.type == 'cpu'
SHAP_MAX_EVALS   = 100 if IS_CPU else 200
LIME_NUM_SAMPLES = 100 if IS_CPU else 200

# Load the fine-tuned model from the local folder if present; otherwise download
# it from the Hugging Face Hub on first run (cached in ~/.cache/huggingface).
HF_REPO_ID = 'code-world/bert-mental-health-detection'
MODEL_SRC  = FT_DIR if os.path.isdir(FT_DIR) else HF_REPO_ID
if MODEL_SRC == HF_REPO_ID:
    print(f"Local model not found — downloading from Hugging Face: {HF_REPO_ID} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_SRC)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_SRC).to(DEVICE).eval()
print(f"Fine-tuned BERT loaded on {DEVICE} (source: {MODEL_SRC}).")


@torch.no_grad()
def predict_proba(texts):
    """Raw texts → BERT → (N, num_classes) probability matrix."""
    texts = [str(t) for t in texts]
    out = []
    for i in range(0, len(texts), 64):
        enc = tokenizer(texts[i:i+64], max_length=128, padding=True,
                        truncation=True, return_tensors='pt')
        logits = model(input_ids=enc['input_ids'].to(DEVICE),
                       attention_mask=enc['attention_mask'].to(DEVICE)).logits
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(out)


shap_explainer = shap.Explainer(predict_proba, shap.maskers.Text(tokenizer),
                                output_names=CLASS_NAMES)
lime_explainer = LimeTextExplainer(class_names=CLASS_NAMES, random_state=42)
print("SHAP Partition explainer + LIME ready.")


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


LEGEND = [mpatches.Patch(color='#2ecc71', label='Pushes toward prediction'),
          mpatches.Patch(color='#e74c3c', label='Pushes away from prediction')]


def bar_chart(words, scores, xlabel, title):
    colours = ['#2ecc71' if s > 0 else '#e74c3c' for s in scores]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(words[::-1], scores[::-1], color=colours[::-1],
            edgecolor='black', alpha=0.85)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11)
    ax.legend(handles=LEGEND, fontsize=8)
    plt.tight_layout()
    return fig


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Mental Health Detection API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class TextInput(BaseModel):
    text: str


@app.get("/")
def health():
    return {"status": "ok", "model": "BERT (fine-tuned)", "classes": CLASS_NAMES}


@app.post("/predict")
def predict(body: TextInput):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    proba      = predict_proba([body.text])[0]
    pred_idx   = int(proba.argmax())
    prediction = CLASS_NAMES[pred_idx]
    confidence = float(proba[pred_idx])
    probs_dict = {c: round(float(p), 4) for c, p in zip(CLASS_NAMES, proba)}
    return {"prediction": prediction, "confidence": round(confidence, 4),
            "probabilities": probs_dict}


@app.post("/explain/lime")
def explain_lime(body: TextInput):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    proba      = predict_proba([body.text])[0]
    pred_idx   = int(proba.argmax())
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(proba[pred_idx])

    exp = lime_explainer.explain_instance(
        body.text, predict_proba, num_features=12,
        num_samples=LIME_NUM_SAMPLES, top_labels=1
    )
    word_weights = exp.as_list(label=exp.top_labels[0])
    words  = [w for w, _ in word_weights]
    scores = [s for _, s in word_weights]

    fig = bar_chart(words, scores, 'LIME Weight',
                    f'LIME Explanation — Predicted: {pred_label} ({confidence:.0%})')
    img = fig_to_base64(fig)
    plt.close(fig)
    return {"image": img, "prediction": pred_label, "confidence": round(confidence, 4)}


@app.post("/explain/shap")
def explain_shap(body: TextInput):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    proba      = predict_proba([body.text])[0]
    pred_idx   = int(proba.argmax())
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(proba[pred_idx])

    sv     = shap_explainer([body.text], max_evals=SHAP_MAX_EVALS, silent=True)
    vals   = sv.values[0][:, pred_idx]
    tokens = np.array([t.strip() for t in sv.data[0]])
    keep   = np.array([len(t) > 0 for t in tokens])
    vals, tokens = vals[keep], tokens[keep]

    top_n   = 12
    top_idx = np.argsort(np.abs(vals))[-top_n:][::-1]
    words   = tokens[top_idx].tolist()
    scores  = vals[top_idx].tolist()

    fig = bar_chart(words, scores, 'SHAP Value',
                    f'SHAP Explanation — Predicted: {pred_label} ({confidence:.0%})')
    img = fig_to_base64(fig)
    plt.close(fig)
    return {"image": img, "prediction": pred_label, "confidence": round(confidence, 4)}
