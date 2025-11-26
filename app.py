import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# ... keep everything above the visualization menu unchanged ...

elif menu == "Visualizations":
    st.header("Visualizations")
    numeric_data = data.select_dtypes(include=['number'])

    if numeric_data.empty:
        st.warning("No numeric columns found for visualization!")
        st.stop()

    chart_type = st.selectbox(
        "Select Chart Type:",
        ["Correlation Heatmap", "Scatter Plot", "Line Graph", "Histogram"]
    )

    # -----------------------
    # Scatter Plot
    # -----------------------
    if chart_type == "Scatter Plot":
        st.subheader("Scatter Plot")
        col1, col2 = st.columns([1, 1])

        with col1:
            x_col = st.selectbox("Select X-axis", numeric_data.columns)

        with col2:
            y_col = st.selectbox("Select Y-axis", numeric_data.columns)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(numeric_data[x_col], numeric_data[y_col])
        ax.set_xlabel(x_col, fontsize=14, labelpad=10)
        ax.set_ylabel(y_col, fontsize=14, labelpad=10)
        ax.set_title(f"{y_col} vs {x_col}", fontsize=16, pad=15)
        ax.grid(True)

        st.pyplot(fig, use_container_width=True)

    # -----------------------
    # Line Graph
    # -----------------------
    elif chart_type == "Line Graph":
        st.subheader("Line Graph")

        line_col = st.selectbox("Select Numeric Column for Line Graph", numeric_data.columns)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(numeric_data[line_col])
        ax.set_xlabel("Index", fontsize=14, labelpad=8)
        ax.set_ylabel(line_col, fontsize=14, labelpad=8)
        ax.set_title(f"Trend of {line_col}", fontsize=16, pad=12)
        ax.grid(True)

        st.pyplot(fig, use_container_width=True)

    # -----------------------
    # Histogram
    # -----------------------
    elif chart_type == "Histogram":
        st.subheader("Histogram")
        hist_col = st.selectbox("Select Numeric Column for Histogram", numeric_data.columns)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(numeric_data[hist_col].dropna(), bins=30)
        ax.set_xlabel(hist_col, fontsize=14, labelpad=10)
        ax.set_ylabel("Frequency", fontsize=14, labelpad=10)
        ax.set_title(f"Distribution of {hist_col}", fontsize=16, pad=15)
        ax.grid(True)

        st.pyplot(fig, use_container_width=True)

    # -----------------------
    # Correlation Heatmap
    # -----------------------
    elif chart_type == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")

        cleaned_data = numeric_data.dropna(axis=1, how="all")
        cleaned_data = cleaned_data.loc[:, cleaned_data.nunique() > 1]

        if cleaned_data.shape[1] < 2:
            st.warning("Not enough valid numeric columns to compute correlation!")
        else:
            corr = cleaned_data.corr()
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap", fontsize=16, pad=12)
            st.pyplot(fig, use_container_width=True)

            # Download matrix
            buffer = BytesIO()
            corr.to_csv(buffer)
            st.download_button("Download Correlation Matrix", buffer.getvalue(), "correlation.csv", "text/csv")

