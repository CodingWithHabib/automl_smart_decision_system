"""
AUTO ML BASED SMART DECISION SUPPORT SYSTEM
Entry point — run with: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="AutoML DSS",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

from modules.upload import render_upload
from modules.eda import render_eda
from modules.preprocessing import render_preprocessing
from modules.training import render_training
from modules.results import render_results
from modules.prediction import render_prediction
from modules.recommendations import render_recommendations

STEPS = [
    "📂 Upload Dataset",
    "🔍 Data Analysis",
    "🧹 Preprocessing",
    "🏋️ Model Training",
    "📊 Results",
    "🎯 Predictions",
    "💡 Recommendations",
]


def init_state():
    defaults = {
        "step": 0,
        "df_raw": None,
        "df_processed": None,
        "target_col": None,
        "problem_type": None,
        "feature_cols": None,
        "models": {},
        "best_model": None,
        "best_model_name": None,
        "metrics": {},
        "label_encoders": {},
        "scaler": None,
        "preprocessing_config": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_sidebar():
    st.sidebar.title("🤖 AutoML DSS")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Progress")
    for i, step in enumerate(STEPS):
        if i < st.session_state.step:
            st.sidebar.markdown(f"✅ {step}")
        elif i == st.session_state.step:
            st.sidebar.markdown(f"**▶️ {step}**")
        else:
            st.sidebar.markdown(f"⬜ {step}")
    st.sidebar.markdown("---")
    if st.session_state.df_raw is not None:
        st.sidebar.markdown("### Dataset Info")
        st.sidebar.write(f"Rows: **{st.session_state.df_raw.shape[0]}**")
        st.sidebar.write(f"Cols: **{st.session_state.df_raw.shape[1]}**")
    if st.session_state.problem_type:
        st.sidebar.write(f"Task: **{st.session_state.problem_type.title()}**")
    if st.session_state.target_col:
        st.sidebar.write(f"Target: **{st.session_state.target_col}**")
    if st.sidebar.button("🔄 Reset Everything"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


def main():
    init_state()
    render_sidebar()

    step = st.session_state.step

    if step == 0:
        render_upload()
    elif step == 1:
        render_eda()
    elif step == 2:
        render_preprocessing()
    elif step == 3:
        render_training()
    elif step == 4:
        render_results()
    elif step == 5:
        render_prediction()
    elif step == 6:
        render_recommendations()

    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if step > 0:
            if st.button("⬅️ Back"):
                st.session_state.step -= 1
                st.rerun()
    with col3:
        can_proceed = (
            (step == 0 and st.session_state.df_raw is not None) or
            (step == 1 and st.session_state.df_raw is not None) or
            (step == 2 and st.session_state.df_processed is not None) or
            (step == 3 and len(st.session_state.models) > 0) or
            (step == 4 and st.session_state.best_model is not None) or
            (step == 5) or
            (step == 6)
        )
        if step == len(STEPS) - 1:
            if st.button("🏠 Start Over"):
                st.session_state.step = 0
                st.rerun()
        elif step < len(STEPS) - 1 and can_proceed:
            if st.button("Next ➡️"):
                st.session_state.step += 1
                st.rerun()


if __name__ == "__main__":
    main()
