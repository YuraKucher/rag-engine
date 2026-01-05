import streamlit as st

from ui.debug.loaders import load_evaluations, load_feedback
from ui.debug.plots import (
    plot_metric_distributions,
    plot_time_series,
    plot_scatter
)


def render_evaluation_dashboard(
    evaluation_path: str,
    feedback_path: str
):
    st.header("📊 Evaluation Dashboard (Debug Mode)")

    df_eval = load_evaluations(evaluation_path)
    df_feedback = load_feedback(feedback_path)

    if df_eval.empty:
        st.warning("No evaluation data found.")
        return

    # Summary
    st.subheader("Summary")
    st.metric("Avg Relevance", round(df_eval["relevance"].mean(), 3))
    st.metric("Avg Groundedness", round(df_eval["groundedness"].mean(), 3))
    st.metric("Avg Answerability", round(df_eval["answerability"].mean(), 3))

    # Plots
    plot_metric_distributions(df_eval)
    plot_time_series(df_eval)
    plot_scatter(df_eval)

    # Feedback (optional)
    if not df_feedback.empty:
        st.subheader("User Feedback")
        st.dataframe(df_feedback)
