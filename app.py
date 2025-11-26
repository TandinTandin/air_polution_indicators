import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.set_page_config(page_title="Air Pollution Dashboard", layout="wide")

st.title("Air Pollution Indicators Dashboard (Bhutan)")

# --------------------------
# Load Data (with fallback warning)
# --------------------------
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv("data/air_pollution_indicators_btn.csv")
        return df
    except FileNotFoundError:
        st.error("Dataset not found! Make sure the CSV is inside the 'data/' folder in your GitHub repo.")
        return None

data = load_data()

if data is None:
    st.stop()

# --------------------------
# Sidebar Menu
# --------------------------
menu = st.sidebar.selectbox(
    "Navigate",
    ["Dataset Overview", "Visualizations"]
)

# --------------------------
# Dataset Overview Section
# --------------------------
if menu == "Dataset Overview":
    st.header("Dataset Overview")
    st.dataframe(data.head(), use_container_width=True)

    st.header("Summary Statistics")
    st.dataframe(data.describe(include="all"), use_container_width=True)

    st.subheader("Column Data Types")
    st.write(data.dtypes)

    categorical_cols = data.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.subheader("Categorical Distributions")
        for col in categorical_cols:
            if data[col].nunique() < 30:  # Avoid large counts
                st.write(f"### {col}")
                st.bar_chart(data[col].value_counts(), use_container_width=True)

# --------------------------
# Visualization Section (All Charts Fixed)
# --------------------------
elif menu == "Visualizations":
    st.header("Visualizations")

    numeric_data = data.select_dtypes(include=["number"])

    if numeric_data.shape[1] < 1:
        st.warning("No numeric columns to visualize!")
        st.stop()

    viz = st.selectbox(
        "Choose Visualization",
        ["Correlation Heatmap", "Scatter Plot", "Line Graph", "Histogram"]
    )

    # ---- Correlation Heatmap ----
    if viz == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")

        corr_method = st.sidebar.selectbox("Select correlation method", ["pearson", "spearman", "kendall"])
        
        cleaned_data = numeric_data.dropna(axis=1, how="all")
        cleaned_data = cleaned_data.loc[:, cleaned_data.nunique() > 1]

        if cleaned_data.shape[1] < 2:
            st.warning("Not enough valid numeric columns for correlation!")
        else:
            corr = cleaned_data.corr(method=corr_method)

            fig, ax = plt.subplots(figsize=(12, 7))
            sns.heatmap(corr, annot=True, fmt=".2f", linewidths=0.5, cbar=True, ax=ax, vmin=-1, vmax=1)
            plt.title(f"Correlation Heatmap ({corr_method.title()})", pad=15)
            st.pyplot(fig, use_container_width=True)

            # Download correlation matrix
            buffer = BytesIO()
            corr.to_csv(buffer, index=True)
            st.download_button("Download Correlation Matrix (CSV)", buffer.getvalue(), "correlation_matrix.csv", "text/csv")

    # ---- Scatter Plot ----
    elif viz == "Scatter Plot":
        st.subheader("Scatter Plot")
        x_col = st.sidebar.selectbox("X-axis", numeric_data.columns)
        y_col = st.sidebar.selectbox("Y-axis", numeric_data.columns)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(numeric_data[x_col], numeric_data[y_col])
        ax.set_title(f"{y_col} vs {x_col}", pad=12)
        ax.set_xlabel(x_col, labelpad=8)
        ax.set_ylabel(y_col, labelpad=8)
        ax.grid(True)

        st.pyplot(fig, use_container_width=True)

    # ---- Line Graph ----
    elif viz == "Line Graph":
        st.subheader("Line Graph")
        col = st.sidebar.selectbox("Select column", numeric_data.columns)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(numeric_data[col])
        ax.set_title(f"Trend of {col}", pad=10)
        ax.set_xlabel("Index", labelpad=6)
        ax.set_ylabel(col, labelpad=6)
        ax.grid(True)

        st.pyplot(fig, use_container_width=True)

    # ---- Histogram ----
    elif viz == "Histogram":
        st.subheader("Histogram")
        col = st.sidebar.selectbox("Select column", numeric_data.columns)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(numeric_data[col].dropna(), bins=25)
        ax.set_title(f"Distribution of {col}", pad=12)
        ax.set_xlabel(col, labelpad=8)
        ax.set_ylabel("Frequency", labelpad=8)
        ax.grid(True)

        st.pyplot(fig, use_container_width=True)

st.caption("EDA and Correlation Dashboard for Bhutan Air Pollution Dataset")
