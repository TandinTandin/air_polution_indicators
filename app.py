import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Air Pollution Indicators Dashboard (Bhutan)")

# Load Dataset

@st.cache_resource
def load_data():
    file_path = "data/air_pollution_indicators_btn.csv"
    df = pd.read_csv(file_path)
    return df

data = load_data()

# Sidebar
menu = st.sidebar.selectbox(
    "Navigate",
    ["Dataset Overview", "Visualizations"]
)

# Dataset Overview
if menu == "Dataset Overview":
    st.header("Dataset Overview")
    st.dataframe(data.head())

    st.header("Summary Statistics")
    st.dataframe(data.describe(include="all"))

    st.header("Column Information")
    st.write(data.dtypes)

    categorical_cols = data.select_dtypes(include=['object']).columns

    if len(categorical_cols) > 0:
        st.subheader("Categorical Distributions")
        for col in categorical_cols:
            if data[col].nunique() < 30:   # avoid huge charts
                st.write(f"### {col}")
                st.bar_chart(data[col].value_counts())

# Visualizations
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

    # Correlation Heatmap (Fixed)
    if viz_type == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")

        # Remove invalid numeric columns
        cleaned_data = numeric_data.dropna(axis=1, how="all")
        cleaned_data = cleaned_data.loc[:, cleaned_data.apply(pd.Series.nunique) > 1]

        if cleaned_data.shape[1] < 2:
            st.warning("Not enough valid numeric columns to generate a correlation heatmap.")
        else:
            corr = cleaned_data.corr()

            plt.figure(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm")
            st.pyplot()

    # Line Chart
    elif viz_type == "Line Chart":
        st.subheader("Line Chart")

        if numeric_data.empty:
            st.warning("No numeric columns available for line chart.")
        else:
            st.line_chart(numeric_data)

    # Bar Chart
    elif viz_type == "Bar Chart":
        column = st.selectbox("Select a column", data.columns)
        st.bar_chart(data[column])

    # Area Chart
    elif viz_type == "Area Chart":
        if numeric_data.empty:
            st.warning("No numeric columns available for area chart.")
        else:
            st.area_chart(numeric_data)

    # Histogram
    elif viz_type == "Histogram":
        if numeric_data.empty:
            st.warning("No numeric columns available for histogram.")
        else:
            feature = st.selectbox("Select a numeric column", numeric_data.columns)
            plt.hist(data[feature].dropna(), bins=20)
            plt.xlabel(feature)
            plt.ylabel("Count")
            st.pyplot()
            
    # Scatter Plot
    elif viz_type == "Scatter Plot":
        if numeric_data.shape[1] < 2:
            st.warning("At least two numeric columns are required for scatter plot.")
        else:
            x_axis = st.selectbox("X-axis:", numeric_data.columns)
            y_axis = st.selectbox("Y-axis:", numeric_data.columns)
            plt.scatter(data[x_axis], data[y_axis])
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            st.pyplot()
