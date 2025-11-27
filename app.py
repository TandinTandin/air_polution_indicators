import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Air Pollution Indicators Dashboard (Bhutan)")

# --------------------------
# Load Dataset
# --------------------------
@st.cache_resource
def load_data():
    file_path = "/mnt/data/air_pollution_indicators_btn.csv"
    df = pd.read_csv(file_path)
    return df

data = load_data()

# --------------------------
# Sidebar Menu
# --------------------------
menu = st.sidebar.selectbox(
    "Navigate",
    ["Dataset Overview", "Visualizations"]
)

# --------------------------
# Dataset Overview
# --------------------------
if menu == "Dataset Overview":
    st.header("Dataset Overview")
    st.dataframe(data.head())

    st.header("Summary Statistics")
    st.dataframe(data.describe(include="all"))

    st.header("Column Information")
    st.write(data.dtypes)

    # Show any categorical value counts
    categorical_cols = data.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.subheader("Categorical Distributions")
        for col in categorical_cols:
            st.write(f"### {col}")
            st.bar_chart(data[col].value_counts())

# --------------------------
# Visualizations
# --------------------------
elif menu == "Visualizations":
    st.header("Visualizations")

    viz_type = st.selectbox(
        "Choose Visualization Type:",
        [
            "Correlation Heatmap",
            "Line Chart",
            "Bar Chart",
            "Area Chart",
            "Histogram",
            "Scatter Plot"
        ]
    )

    numeric_data = data.select_dtypes(include=['number'])

    # Correlation Heatmap
    if viz_type == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        corr = numeric_data.corr()

        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        st.pyplot()

    # Line Chart
    elif viz_type == "Line Chart":
        st.subheader("Line Chart")
        st.line_chart(numeric_data)

    # Bar Chart
    elif viz_type == "Bar Chart":
        col = st.selectbox("Select a column", data.columns)
        st.bar_chart(data[col])

    # Area Chart
    elif viz_type == "Area Chart":
        st.area_chart(numeric_data)

    # Histogram
    elif viz_type == "Histogram":
        feature = st.selectbox("Select a numeric column", numeric_data.columns)
        plt.hist(data[feature], bins=20)
        plt.xlabel(feature)
        plt.ylabel("Count")
        st.pyplot()

    # Scatter Plot
    elif viz_type == "Scatter Plot":
        x_axis = st.selectbox("X-axis:", numeric_data.columns)
        y_axis = st.selectbox("Y-axis:", numeric_data.columns)
        plt.scatter(data[x_axis], data[y_axis])
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        st.pyplot()
