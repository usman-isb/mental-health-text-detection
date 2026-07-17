#!/usr/bin/env bash
# Start the FastAPI backend, then the Streamlit frontend (the published UI port).
set -e

uvicorn main:app --app-dir webapp/backend --host 0.0.0.0 --port 8000 &

# Tell the frontend to use the API instead of loading the model in-process.
export MINDSCAN_API=http://localhost:8000

exec streamlit run webapp/frontend/app.py \
    --server.address 0.0.0.0 \
    --server.port 8501
