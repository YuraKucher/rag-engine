import streamlit as st
import pandas as pd


def plot_metric_distributions(df: pd.DataFrame):
    st.subheader("Metric Distributions")

    for metric in ["relevance", "groundedness", "answerability"]:
        if metric in df:
            st.markdown(f"**{metric.capitalize()}**")
            st.bar_chart(df[metric].dropna())


def plot_time_series(df: pd.DataFrame):
    st.subheader("Quality Over Time")

    df_sorted = df.sort_values("created_at")
    ts = df_sorted.set_index("created_at")[[
        "relevance", "groundedness", "answerability"
    ]]

    st.line_chart(ts)


def plot_scatter(df: pd.DataFrame):
    st.subheader("Metric Relationships")

    if "relevance" in df and "groundedness" in df:
        st.scatter_chart(
            df,
            x="relevance",
            y="groundedness"
        )
