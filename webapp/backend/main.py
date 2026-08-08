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


GREEN_HL, RED_HL = '#2E9E63', '#D1495B'

# Filler words carry no clinical meaning. Negations are deliberately kept —
# "not", "never", "cannot" change the meaning of a sentence entirely.
FILLER = {
    'a', 'an', 'the', 'and', 'or', 'but', 'so', 'then', 'than', 'as', 'if',
    'in', 'on', 'at', 'of', 'to', 'for', 'from', 'with', 'by', 'about', 'into',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
    'can', 'could', 'may', 'might', 'must', 'shall',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'them',
    'us', 'my', 'your', 'his', 'its', 'our', 'their', 'this', 'that', 'these',
    'those', 'there', 'here', 'what', 'which', 'who', 'when', 'where', 'how',
    'just', 'really', 'very', 'also', 'too', 'much', 'many', 'some', 'any',
    'thing', 'things', 'get', 'got', 'go', 'going', 'like', 'one', 'now',
    'up', 'out', 'down', 'over', 'again', 'still', 'even', 'ever',
    's', 't', 'm', 're', 've', 'll', 'd',
}
KEEP_ALWAYS = {'not', 'no', 'never', 'cannot', 'nothing', 'nobody', 'none'}

W_IN, MARGIN, FIG_DPI, FS_BODY, ROW_IN = 9.6, 0.30, 130, 14, 0.34
TOP_N = 6


def _key(w):
    return str(w).lower().strip('.,!?;:\'"()[]-#')


def _rank_tokens(pairs, top_n=TOP_N):
    """Strongest meaningful tokens first, de-duplicated."""
    ranked, seen = [], set()
    for w, s in pairs:
        k = _key(w)
        if not k or k in seen:
            continue
        if k not in KEEP_ALWAYS and (k in FILLER or len(k) < 3
                                     or not any(c.isalpha() for c in k)):
            continue
        seen.add(k)
        ranked.append((k, float(s)))
    ranked.sort(key=lambda p: abs(p[1]), reverse=True)
    return ranked[:top_n]


def _weight_for(word, weights):
    """Exact match first, then sub-word match (SHAP emits token fragments)."""
    k = _key(word)
    if k in weights:
        return weights[k]
    for cand, s in weights.items():
        if len(cand) >= 4 and cand in k:
            return s
    return None


def _measure_words(words, weights):
    fig = plt.figure(figsize=(W_IN, 1), dpi=FIG_DPI)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
    fig.canvas.draw(); rend = fig.canvas.get_renderer()
    out = {}
    for w in set(words) | {' '}:
        bold = 'bold' if _weight_for(w, weights) is not None else 'normal'
        t = ax.text(0, 0, w, fontsize=FS_BODY, fontweight=bold)
        out[w] = t.get_window_extent(rend).width / FIG_DPI
        t.remove()
    plt.close(fig)
    return out


def _highlight_figure(text, pairs, pred_label, confidence, method):
    """Render the post itself with the evidence words highlighted."""
    top = _rank_tokens(pairs)
    weights = {k: s for k, s in top}
    maxabs = max((abs(s) for s in weights.values()), default=1.0) or 1.0

    words = text.split()
    widths = _measure_words(words, weights)
    space, usable = widths[' '], W_IN - 2 * MARGIN

    lines, cur, x = [], [], 0.0
    for w in words:
        ww = widths[w]
        if x + ww > usable and cur:
            lines.append(cur); cur, x = [], 0.0
        cur.append((w, x, ww)); x += ww + space
    if cur:
        lines.append(cur)

    h_in = (2.9 + len(lines)) * ROW_IN
    fig = plt.figure(figsize=(W_IN, h_in), dpi=FIG_DPI)
    ax = fig.add_axes([MARGIN / W_IN, 0.02, usable / W_IN, 0.96])
    ax.axis('off'); ax.set_xlim(0, usable); ax.set_ylim(0, h_in)

    y = h_in - 0.34
    ax.text(0, y, f'Why the model predicted {pred_label} ({confidence:.0%})',
            fontsize=15.5, fontweight='bold', va='top')
    y -= 0.34
    ax.text(0, y, 'Green = words supporting this prediction   ·   red = words '
                  f'against it   ·   stronger colour = more important   ·   {method}',
            fontsize=10, color='#666666', va='top')

    y -= 0.42
    for line in lines:
        for w, x0, ww in line:
            s = _weight_for(w, weights)
            if s is not None:
                alpha = 0.32 + 0.58 * (abs(s) / maxabs)
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x0 - 0.03, y - 0.105), ww + 0.06, 0.22,
                    boxstyle='round,pad=0.01,rounding_size=0.04',
                    fc=GREEN_HL if s > 0 else RED_HL, ec='none',
                    alpha=alpha, zorder=0))
            ax.text(x0, y, w, fontsize=FS_BODY, va='center', zorder=2,
                    fontweight='bold' if s is not None else 'normal')
        y -= ROW_IN

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

    fig = _highlight_figure(body.text,
                            word_weights, pred_label, confidence, 'LIME')
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

    fig = _highlight_figure(body.text,
                            list(zip(tokens.tolist(), vals.tolist())),
                            pred_label, confidence, 'SHAP')
    img = fig_to_base64(fig)
    plt.close(fig)
    return {"image": img, "prediction": pred_label, "confidence": round(confidence, 4)}
