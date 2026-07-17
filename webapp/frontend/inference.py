"""
In-process model service — used when the frontend runs without the FastAPI
backend (e.g. on Streamlit Community Cloud, where only one process runs).
Mirrors the prediction/explanation logic and response shapes of
webapp/backend/main.py so analyser.py can use either interchangeably.
"""

import base64
import io
import pickle
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import torch
import shap
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings('ignore')

# repo layout: data/webapp/frontend/inference.py → data/
BASE       = Path(__file__).resolve().parents[2]
SPLITS_DIR = BASE / 'datasets' / 'processed' / 'splits'
FT_DIR     = BASE / 'models' / 'saved' / 'bert_finetuned'

# Same fallback as the backend: local checkpoint if present, otherwise download
# from the Hugging Face Hub on first run (cached in ~/.cache/huggingface).
HF_REPO_ID = 'code-world/bert-mental-health-detection'


@st.cache_resource(show_spinner="Loading fine-tuned BERT — first run downloads ~440 MB…")
def _service():
    with open(SPLITS_DIR / 'label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    class_names = list(le.classes_)

    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    is_cpu = device.type == 'cpu'

    model_src = str(FT_DIR) if FT_DIR.is_dir() else HF_REPO_ID
    tokenizer = AutoTokenizer.from_pretrained(model_src)
    model     = (AutoModelForSequenceClassification.from_pretrained(model_src)
                 .to(device).eval())

    @torch.no_grad()
    def predict_proba(texts):
        """Raw texts → BERT → (N, num_classes) probability matrix."""
        texts = [str(t) for t in texts]
        out = []
        for i in range(0, len(texts), 64):
            enc = tokenizer(texts[i:i+64], max_length=128, padding=True,
                            truncation=True, return_tensors='pt')
            logits = model(input_ids=enc['input_ids'].to(device),
                           attention_mask=enc['attention_mask'].to(device)).logits
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.vstack(out)

    return {
        'class_names':   class_names,
        'predict_proba': predict_proba,
        'shap':          shap.Explainer(predict_proba, shap.maskers.Text(tokenizer),
                                        output_names=class_names),
        'lime':          LimeTextExplainer(class_names=class_names, random_state=42),
        # Each explanation needs hundreds of BERT forward passes — use smaller
        # sampling budgets on CPU-only machines to keep response times sane.
        'shap_max_evals':   100 if is_cpu else 200,
        'lime_num_samples': 100 if is_cpu else 200,
    }


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


LEGEND = [mpatches.Patch(color='#2ecc71', label='Pushes toward prediction'),
          mpatches.Patch(color='#e74c3c', label='Pushes away from prediction')]


def _bar_chart(words, scores, xlabel, title):
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


def predict(text: str) -> dict:
    svc        = _service()
    proba      = svc['predict_proba']([text])[0]
    pred_idx   = int(proba.argmax())
    prediction = svc['class_names'][pred_idx]
    confidence = float(proba[pred_idx])
    probs_dict = {c: round(float(p), 4)
                  for c, p in zip(svc['class_names'], proba)}
    return {"prediction": prediction, "confidence": round(confidence, 4),
            "probabilities": probs_dict}


def explain_lime(text: str) -> dict:
    svc        = _service()
    proba      = svc['predict_proba']([text])[0]
    pred_idx   = int(proba.argmax())
    pred_label = svc['class_names'][pred_idx]
    confidence = float(proba[pred_idx])

    exp = svc['lime'].explain_instance(
        text, svc['predict_proba'], num_features=12,
        num_samples=svc['lime_num_samples'], top_labels=1
    )
    word_weights = exp.as_list(label=exp.top_labels[0])
    words  = [w for w, _ in word_weights]
    scores = [s for _, s in word_weights]

    fig = _bar_chart(words, scores, 'LIME Weight',
                     f'LIME Explanation — Predicted: {pred_label} ({confidence:.0%})')
    img = _fig_to_base64(fig)
    plt.close(fig)
    return {"image": img, "prediction": pred_label, "confidence": round(confidence, 4)}


def explain_shap(text: str) -> dict:
    svc        = _service()
    proba      = svc['predict_proba']([text])[0]
    pred_idx   = int(proba.argmax())
    pred_label = svc['class_names'][pred_idx]
    confidence = float(proba[pred_idx])

    sv     = svc['shap']([text], max_evals=svc['shap_max_evals'], silent=True)
    vals   = sv.values[0][:, pred_idx]
    tokens = np.array([t.strip() for t in sv.data[0]])
    keep   = np.array([len(t) > 0 for t in tokens])
    vals, tokens = vals[keep], tokens[keep]

    top_n   = 12
    top_idx = np.argsort(np.abs(vals))[-top_n:][::-1]
    words   = tokens[top_idx].tolist()
    scores  = vals[top_idx].tolist()

    fig = _bar_chart(words, scores, 'SHAP Value',
                     f'SHAP Explanation — Predicted: {pred_label} ({confidence:.0%})')
    img = _fig_to_base64(fig)
    plt.close(fig)
    return {"image": img, "prediction": pred_label, "confidence": round(confidence, 4)}
