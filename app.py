import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.set_page_config(page_title="Air Pollution Dashboard", layout="wide")

st.title("Air Pollution Indicators Dashboard (Bhutan)")

# --------------------------
# Load Dataset with Upload Support
# --------------------------
@st.cache_resource
def load_data_local(path):
    try:
        return pd.read_csv(path)
    except:
        return None

uploaded_file = st.file_uploader("Upload CSV (Optional)", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("CSV Uploaded Successfully!")
else:
    local_path = "data/air_pollution_indicators_btn.csv"
    data = load_data_local := load_data_local(local_path)
    if load_data_local is None:
        st.warning("No CSV uploaded and local file not found in repo. Please upload the dataset.")
        st.stop()
    else:
        data = load_data_local

# --------------------------
# Sidebar Navigation
# --------------------------
menu = st.sidebar.selectbox("Navigate", ["Dataset Overview", "Visualizations"])

# --------------------------
# Dataset Overview
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
            if data[col].nunique() < 30:
                st.write(f"### {col}")
                st.bar_chart(data[col].value_counts(), use_container_width=True)

# --------------------------
# Visualizations
# --------------------------
elif menu == "Visualizations":
    st.header("Visualizations")

    viz_type = st.selectbox(
        "Choose Visualization Type:",
        ["Correlation Heatmap", "Line Chart", "Bar Chart", "Area Chart", "Histogram", "Scatter Plot"]
    )

    numeric_data = data.select_dtypes(include=['number'])

    # --------------------------
    # Correlation Heatmap (Improved)
    # --------------------------
    if viz_type == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")

        corr_method = st.sidebar.selectbox("Correlation Method", ["pearson", "spearman", "kendall"])
        cleaned_data = numeric_data.dropna(axis=1, how="all")
        cleaned_data = cleaned_data.loc[:, cleaned_data.apply(pd.Series.nunique) > 1]

        if cleaned_data.shape[1] < 2:
            st.warning("Not enough valid numeric columns for correlation.")
        else:
            corr = cleaned_data.corr(method=corr_method)

            # Heatmap
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                linewidths=0.4,
                cbar=True,
                vmin=-1,
                vmax=1,
                ax=ax
            )
            plt.title(f"Correlation Heatmap ({corr_method.title()})", pad=15)
            st.pyplot(fig, use_container_width=True)

            # Download Correlation Matrix
            buffer = BytesIO()
            corr.to_csv(buffer)
            st.download_button("Download Correlation Matrix", buffer.getvalue(), "correlation.csv", "text/csv")

    # --------------------------
    # Line Chart
    # --------------------------
    elif viz_type == "Line Chart":
        st.subheader("Line Chart")
        if numeric_data.empty:
            st.warning("No numeric data available.")
        else:
            st.line_chart(numeric_data, use_container_width=True)

    # --------------------------
    # Bar Chart (improved labeling)
    # --------------------------
    elif viz_type == "Bar Chart":
        col = st.sidebar.selectbox("Select Column", data.columns)
        if data[col].dtype not in ['int64','float64']:
            st.warning("Bar chart works best with numeric columns.")
        st.bar_chart(data[col].value_counts() if data[col].dtype=='object' else data[col], use_container_width=True)

    # --------------------------
    # Area Chart
    # --------------------------
    elif viz_type == "Area Chart":
        st.subheader("Area Chart")
        if numeric_data.empty:
            st.warning("No numeric data available.")
        else:
            st.area_chart(numeric_data, use_container_width=True)

    # --------------------------
    # Histogram with better Aesthetic
    # --------------------------
    elif viz_type == "Histogram":
        st.subheader("Histogram")
        if numeric_data.empty:
            st.warning("No numeric data available.")
        else:
            feature = st.sidebar.selectbox("Select Numeric Feature", numeric_data.columns)

            fig, ax = plt.subplots(figsize=(10,5))
            ax.hist(numeric_data[feature].dropna(), bins=25)
            ax.set_xlabel(feature, labelpad=10, fontsize=14)
            ax.set_ylabel("Frequency", labelpad=10, fontsize=14)
            ax.set_title(f"Distribution of {feature}", pad=12, fontsize=16)
            st.pyplot(fig, use_container_width=True)

    # --------------------------
    # Scatter Plot (Improved)
    # --------------------------
    elif viz_type == "Scatter Plot":
        st.subheader("Scatter Plot")
        if numeric_data.shape[1] < 2:
            st.warning("At least 2 numeric columns needed.")
        else:
            x = st.sidebar.selectbox("X-axis", numeric_data.columns)
            y = st.sidebar.selectbox("Y-axis", numeric_data.columns)

            fig, ax = plt.subplots(figsize=(10,6))
            ax.scatter(numeric_data[x], numeric_data[y])
            ax.set_xlabel(x, fontsize=13, labelpad=8)
            ax.set_ylabel(y, fontsize=13, labelpad=8)
            ax.set_title(f"{y} vs {x}", fontsize=15, pad=12)
            ax.grid(True)
            st.pyplot(fig, use_container_width=True)

st.caption("Correlation and EDA Dashboard for Air Pollution Dataset — Bhutan")
