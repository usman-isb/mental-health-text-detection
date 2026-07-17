# Hugging Face Spaces (Docker SDK) — root Dockerfile required by Spaces.
# Local development keeps using webapp/Dockerfile via docker-compose.yml.
#
# The fine-tuned model is NOT baked into the image: on first start the backend
# downloads it from the Hub (code-world/bert-mental-health-detection) and
# caches it under HF_HOME.
FROM python:3.11-slim

# Spaces run the container as a non-root user (UID 1000) — create it and give
# it writable cache/config locations.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# CPU-only torch wheel (Spaces free tier has no GPU)
COPY webapp/requirements-docker.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-docker.txt

# App code plus the small artefacts that docker-compose normally mounts as
# volumes — Spaces has no volume mounts, so they must live in the image.
COPY --chown=user webapp ./webapp
COPY --chown=user results ./results
COPY --chown=user datasets/processed/splits/label_encoder.pkl ./datasets/processed/splits/label_encoder.pkl

USER user

# Streamlit UI — the single public port (declared as app_port in README.md).
# FastAPI runs on localhost:8000 inside the container, reachable only by the UI.
EXPOSE 8501
CMD ["bash", "webapp/start-docker.sh"]
