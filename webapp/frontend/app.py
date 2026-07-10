"""
Streamlit Frontend — Mental Health Detection
Entry point / page router. Calls the FastAPI backend at http://localhost:8000
Run: streamlit run webapp/frontend/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="MindScan — Mental Health Text Analyser",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation([
    st.Page("analyser.py",     title="Analyser",         icon="🧠", default=True),
    st.Page("results_page.py", title="Results & EDA",    icon="📊"),
    st.Page("project_page.py", title="Project Overview", icon="📋"),
])
pg.run()
