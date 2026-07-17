FROM python:3.11-slim

WORKDIR /app

# CPU-only torch wheel (containers have no GPU access on macOS)
COPY webapp/requirements-docker.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-docker.txt

COPY webapp ./webapp

ENV MPLCONFIGDIR=/tmp/matplotlib \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8000 8501
CMD ["bash", "webapp/start-docker.sh"]
