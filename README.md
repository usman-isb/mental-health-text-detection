# Explainable Transformer Based Framework for Early Detection of Mental Health Disorders from Social Media Text

COM748 Masters Research Project — Miraj

---

## What this project does

Detects mental health conditions from social media text across 5 classes:
**Anxiety/Stress · Bipolar · Depression · Normal · Suicidal**

Compares ML, Deep Learning, and Transformer (BERT) models, then deploys the best model in a web application with SHAP and LIME explanations.

---

## Requirements

- Python **3.10, 3.11, 3.12, or 3.13** (Apple Silicon Macs / Windows / Linux)
  — on **Intel Macs use 3.10–3.12** (PyTorch has no Intel-Mac wheels for 3.13)
- A Kaggle account (free) with an API key
- ~4 GB free disk space
- Internet connection (first run only, to download datasets and BERT weights)

---

## Setup — macOS

### 1. Install Python

Download the macOS installer from https://www.python.org/downloads/

- **Apple Silicon Mac (M1/M2/M3/M4):** install version **3.13.x** (or 3.10–3.12)
- **Intel Mac (e.g. 2017 MacBook Pro):** install version **3.12.x** (or 3.10/3.11) —
  do **not** use 3.13, PyTorch does not support it on Intel Macs.
  `requirements.txt` automatically installs the Intel-compatible library versions
  (torch 2.2.2, transformers 4.x, numpy 1.x).

To check which Mac you have: Apple menu →  About This Mac — "Chip: Apple M…" is
Apple Silicon; "Processor: … Intel Core …" is Intel.

Verify in Terminal:
```
python3 --version
```

### 2. Clone / copy the project

Place the project folder anywhere on your machine, e.g. `~/Desktop/project/data`.

Open Terminal and navigate into it:
```
cd ~/Desktop/project/data
```

### 3. Create a virtual environment

```
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### 4. Install dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

This installs all required libraries (PyTorch, XGBoost, scikit-learn, Transformers, FastAPI, Streamlit, SHAP, LIME, etc.).

### 5. Set up Kaggle API key

1. Log in to https://www.kaggle.com
2. Go to your profile → Settings → API → Create New Token
3. A file called `kaggle.json` will download
4. Move it to `~/.kaggle/kaggle.json`:

```
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

---

## Setup — Windows

### 1. Install Python

Download the Windows installer from https://www.python.org/downloads/  
Install version **3.13.x** (or 3.10–3.12).

**Important:** tick "Add Python to PATH" during installation.

Verify in Command Prompt:
```
python --version
```

### 2. Clone / copy the project

Place the project folder anywhere, e.g. `C:\Users\YourName\Desktop\project\data`.

Open Command Prompt and navigate into it:
```
cd C:\Users\YourName\Desktop\project\data
```

### 3. Create a virtual environment

```
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your prompt.

### 4. Install dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Set up Kaggle API key

1. Log in to https://www.kaggle.com
2. Go to your profile → Settings → API → Create New Token
3. A file called `kaggle.json` will download
4. Move it to `C:\Users\YourName\.kaggle\kaggle.json`
   - Create the `.kaggle` folder if it does not exist

---

## Running the pipeline

All steps run as Jupyter notebooks. Start Jupyter with:

**macOS:**
```
source venv/bin/activate
venv/bin/jupyter notebook notebooks/
```

**Windows:**
```
venv\Scripts\activate
venv\Scripts\jupyter notebook notebooks\
```

Run the notebooks in order:

| Notebook | What it does |
|---|---|
| `00_download_datasets.ipynb` | Downloads the two Kaggle datasets |
| `01_explore_datasets.ipynb` | Explores class distribution and text length |
| `02_preprocess.ipynb` | Cleans text, merges datasets, splits train/val/test |
| `03_features.ipynb` | Builds TF-IDF feature matrices |
| `04_train_ml.ipynb` | Trains Logistic Regression and XGBoost |
| `05_train_dl.ipynb` | Trains CNN and BiLSTM |
| `06_train_bert.ipynb` | Fine-tunes BERT |
| `07_compare.ipynb` | Compares all models |
| `08_explainability.ipynb` | SHAP and LIME explanations |

Each notebook saves its outputs (models, processed data) automatically. You do not need to run them all at once.

---

## Running the web application

The webapp has two parts: a FastAPI backend and a Streamlit frontend.

### Start the backend

**macOS:**
```
source venv/bin/activate
cd webapp/backend
uvicorn main:app --reload --port 8000
```

**Windows:**
```
venv\Scripts\activate
cd webapp\backend
uvicorn main:app --reload --port 8000
```

### Start the frontend (new terminal window)

**macOS:**
```
source venv/bin/activate
streamlit run webapp/frontend/app.py
```

**Windows:**
```
venv\Scripts\activate
streamlit run webapp\frontend\app.py
```

Open your browser at **http://localhost:8501**

---

## Project structure

```
data/
  datasets/
    raw/           ← downloaded CSV files
    processed/     ← cleaned data, TF-IDF matrices, train/val/test splits
  models/
    saved/         ← trained model files (.pkl, .json, .pt)
  notebooks/       ← pipeline notebooks (run in order)
  webapp/
    backend/       ← FastAPI prediction API
    frontend/      ← Streamlit UI
  requirements.txt
  README.md
```

---

## Troubleshooting

**`kaggle: command not found`** — make sure your virtual environment is activated.

**`ModuleNotFoundError`** — run `pip install -r requirements.txt` again with the venv activated.

**Slow training on Windows** — XGBoost and PyTorch run on CPU by default. Training may take 5–15 minutes per model. This is normal.

**BERT download on first run** — `06_train_bert.ipynb` downloads ~440 MB of model weights from HuggingFace on first execution. This is automatic but requires internet. (The web app does the same on its own first start — see `webapp/README.md`.)
