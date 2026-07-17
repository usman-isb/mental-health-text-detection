#!/usr/bin/env bash
# Deploy ONLY the web application to a Hugging Face Space (Docker SDK).
# The Space gets: Dockerfile, webapp/, results/, label_encoder.pkl and a small
# README — none of the notebooks, datasets or training code.
#
# One-time prerequisites:
#   1. Create the Space at https://huggingface.co/new-space
#        Owner: code-world · Space SDK: Docker · Hardware: CPU basic (free)
#   2. Log in so git can push:
#        pip install -U huggingface_hub
#        hf auth login --add-to-git-credential
#
# Usage:  bash deploy_hf_space.sh [owner/space-name]   (default: code-world/mindscan)
set -euo pipefail

SPACE_ID="${1:-code-world/mindscan}"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "Staging web-app-only files for $SPACE_ID ..."
mkdir -p "$STAGE/datasets/processed/splits"
cp "$REPO_DIR/Dockerfile" "$STAGE/"
cp -R "$REPO_DIR/webapp" "$REPO_DIR/results" "$STAGE/"
cp "$REPO_DIR/datasets/processed/splits/label_encoder.pkl" \
   "$STAGE/datasets/processed/splits/"

# strip editor/cache junk from the staged copy
find "$STAGE" \( -name '__pycache__' -o -name '.idea' -o -name '.DS_Store' \) \
     -exec rm -rf {} + 2>/dev/null || true

# Space README — the YAML front-matter is what tells Spaces how to run the app
cat > "$STAGE/README.md" <<'EOF'
---
title: MindScan — Mental Health Detection
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
---

# MindScan — Explainable Mental Health Detection from Text

Streamlit UI + FastAPI backend serving a fine-tuned BERT classifier
(Anxiety/Stress, Bipolar, Depression, Normal, Suicidal) with SHAP and LIME
explanations. The model is downloaded on first start from
[code-world/bert-mental-health-detection](https://huggingface.co/code-world/bert-mental-health-detection).
EOF

cd "$STAGE"
git init -q -b main
git add -A
git -c user.name="Usman" -c user.email="usman.bsse884@gmail.com" \
    commit -qm "Deploy web app"
git push --force "https://huggingface.co/spaces/$SPACE_ID" main

echo ""
echo "Done. The Space is building now (first build ~5-10 min):"
echo "  https://huggingface.co/spaces/$SPACE_ID"
