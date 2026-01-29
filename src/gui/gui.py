import streamlit as st
from pathlib import Path


st.set_page_config(
    layout="wide",
    page_title="ML Sandbox",
    initial_sidebar_state='collapsed'
)
root = Path(__file__).parent

pages = [
    st.Page(
        str(root / "load_data.py"),
        title="Daten upload",
        icon=":material/dataset:"
    ),
    st.Page(
        str(root / "stats.py"),
        title="Stats",
        icon=":material/search_insights:",
    ),
    st.Page(
        str(root / "machine_learning.py"),
        title="ML",
        icon=":material/model_training:",
    ),
    st.Page(
        str(root / "results.py"),
        title="Datenvisualisierung",
        icon=":material/show_chart:",
    )
]
# st.sidebar.image(
    # "../assets/logo_informu.svg", use_container_width=False, width=70
# )


pg = st.navigation(pages)
pg.run()
