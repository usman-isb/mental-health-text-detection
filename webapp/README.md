# MindScan — Mental Health Detection Web App

Web application for the project *Explainable Transformer Based Framework for Early Detection
of Mental Health Disorders from Social Media Text* (COM748 Masters Research Project).

Classifies a social media post into one of **5 classes** — Anxiety/Stress, Bipolar, Depression,
Normal, Suicidal — using a **fine-tuned BERT** model (test accuracy 83.4%, macro-F1 0.841),
and explains every prediction with **LIME** and **SHAP**.

## Architecture

```
Streamlit frontend (port 8501)  ──HTTP──▶  FastAPI backend (port 8000)  ──▶  Fine-tuned BERT
  🧠 Analyser                                POST /predict                     (local folder, or
  📊 Results & EDA                           POST /explain/lime                 auto-downloaded from
  📋 Project Overview                        POST /explain/shap                 Hugging Face Hub)
```

## The model

The backend loads the fine-tuned model from `models/saved/bert_finetuned/` if it exists
(produced by `notebooks/06_train_bert.ipynb`). **If the folder is missing, the model is
downloaded automatically from Hugging Face on first startup** (~440 MB, cached afterwards
in `~/.cache/huggingface`):

> https://huggingface.co/usman-isb/bert-mental-health-detection

No token is needed — the repository is public. So you can run the web app on a fresh machine
without ever running the training notebooks.

## Requirements

- Python with the project virtual environment set up (see the main project
  [README](../README.md) for `venv` setup and `pip install -r requirements.txt`)
  - **Apple Silicon Mac / Windows / Linux:** Python 3.10 – 3.13
  - **Intel Mac (e.g. 2017 MacBook Pro):** Python **3.10 – 3.12** — PyTorch no longer
    publishes Intel-Mac wheels for newer Pythons; `requirements.txt` automatically selects
    the compatible versions (`torch 2.2.2`, `transformers 4.x`, `numpy 1.x`) on Intel Macs
- ~1 GB free disk space (model download + cache) on first run
- 8 GB RAM is sufficient (the model needs ~1.5 GB while running)
- The label encoder must exist: `datasets/processed/splits/label_encoder.pkl`
  (created by `notebooks/02_preprocess.ipynb`; already included if you copied the project folder)

## Run (macOS)

Open two terminals in the project root (the folder containing `webapp/`).

**Terminal 1 — backend:**

```bash
source venv/bin/activate
cd webapp/backend
uvicorn main:app --port 8000
```

**Terminal 2 — frontend:**

```bash
source venv/bin/activate
cd webapp/frontend
streamlit run app.py
```

Device selection is automatic: Apple-Silicon Macs use the GPU (MPS); **Intel Macs run on
CPU** — predictions take a few seconds and LIME/SHAP explanations up to a minute or two
(the backend automatically uses smaller sampling budgets on CPU to compensate).

## Run (Windows)

Open two terminals (Command Prompt or PowerShell) in the project root (the folder
containing `webapp/`).

**Terminal 1 — backend:**

```bat
venv\Scripts\activate
cd webapp\backend
uvicorn main:app --port 8000
```

**Terminal 2 — frontend:**

```bat
venv\Scripts\activate
cd webapp\frontend
streamlit run app.py
```

On Windows the device is NVIDIA GPU (CUDA) if available, otherwise CPU.

### First run (all platforms)

Wait for `Fine-tuned BERT loaded on <device>` in the backend log before using the app.
On first run you will see `Local model not found — downloading from Hugging Face ...`
— the model (~440 MB) is downloaded once and cached in `~/.cache/huggingface`
(`C:\Users\<name>\.cache\huggingface` on Windows).

Then open **http://localhost:8501** in a browser.

## Pages

| Page | What it shows |
|---|---|
| 🧠 Analyser | Paste a post → prediction, class probabilities, LIME and SHAP explanations |
| 📊 Results & EDA | Full experimental results: EDA, 6-model comparison, fine-tuning curves, confusion matrices, explainability analysis |
| 📋 Project Overview | Aim, objectives, methodology, datasets, architecture, ethics and future work |

## API (backend)

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| `/` | GET | — | Health check + model name |
| `/predict` | POST | `{"text": "..."}` | Prediction, confidence, per-class probabilities |
| `/explain/lime` | POST | `{"text": "..."}` | LIME chart (base64 PNG) + prediction |
| `/explain/shap` | POST | `{"text": "..."}` | SHAP chart (base64 PNG) + prediction |

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "lately I cannot stop worrying about everything, my heart races all day"}'
```

## Troubleshooting

- **"Backend offline" in the UI** — the backend is not running or still loading the model;
  check Terminal 1.
- **Slow SHAP/LIME explanations** — each explanation runs 100–200 BERT forward passes
  (100 on CPU-only machines, 200 on GPU); the first request is slowest, later ones are
  faster once the model is warm. On a dual-core Intel Mac expect roughly 30–90 seconds.
- **`pip install` fails on an Intel Mac** — make sure the venv uses Python 3.10–3.12
  (`python3 --version`), not 3.13; then re-run `pip install -r requirements.txt`.
- **Download fails on first run** — check the internet connection; the model is fetched from
  the public Hugging Face repository listed above.

---

⚠️ **Research prototype only.** Not a clinical diagnostic instrument. Do not use as a
substitute for professional mental health assessment.
