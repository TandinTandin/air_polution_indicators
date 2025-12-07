import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Air Pollution Indicators Dashboard (Bhutan)")

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
@st.cache_resource
def load_data():
    file_path = "data/air_pollution_indicators_btn.csv"  # adjust path as needed
    df = pd.read_csv(file_path)
    return df

data = load_data()

# Convert key numeric fields
data["Value"] = pd.to_numeric(data["Value"], errors="coerce")
data["Low"] = pd.to_numeric(data["Low"], errors="coerce")
data["High"] = pd.to_numeric(data["High"], errors="coerce")
if "STARTYEAR" in data.columns:
    data["STARTYEAR"] = pd.to_numeric(data["STARTYEAR"], errors="coerce")

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
menu = st.sidebar.selectbox(
    "Navigate",
    ["Dataset Overview", "Visualizations"]
)

# ---------------------------------------------------------
# Dataset Overview Section
# ---------------------------------------------------------
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
            if data[col].nunique() < 30:
                st.write(f"### {col}")
                st.bar_chart(data[col].value_counts())

# ---------------------------------------------------------
# Visualization Section
# ---------------------------------------------------------
elif menu == "Visualizations":
    st.header("📊 Pollution Visualizations (Bhutan)")

    viz_type = st.selectbox(
        "Choose a visualization:",
        [
            "Pollution Trend Over Time",
            "Pollution Type Comparison",
            "Region-wise Pollution Levels",
            "Pollution Range View",
        ]
    )

    # ---------------------------------------------------------
    # 📈 1. Pollution Trend Over Time
    # ---------------------------------------------------------
    if viz_type == "Pollution Trend Over Time":
        st.subheader("📈 Pollution Trend Over Time")

        indicator_list = data["GHO (DISPLAY)"].dropna().unique()
        selected_indicator = st.selectbox("Select Indicator:", indicator_list)

        filtered = data[data["GHO (DISPLAY)"] == selected_indicator]
        filtered = filtered.dropna(subset=["STARTYEAR", "Value"])

        if filtered.empty:
            st.error("No data available for this indicator.")
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(filtered["STARTYEAR"], filtered["Value"], marker="o")
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.set_title(f"Trend Over Time: {selected_indicator}")
            plt.tight_layout()
            st.pyplot(fig)

    # ---------------------------------------------------------
    # 📊 2. Pollution Type Comparison (Improved Bar Chart)
    # ---------------------------------------------------------
    elif viz_type == "Pollution Type Comparison":
        st.subheader("📊 Indicator Comparison (Top 10) — Improved")

        indicator_avg = (
            data.groupby("GHO (DISPLAY)")["Value"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(indicator_avg.index, indicator_avg.values)
        ax.set_xlabel("Average Value", fontsize=12)
        ax.set_title("Top 10 Indicators by Average Value", fontsize=14)
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

    # ---------------------------------------------------------
    # 🌍 3. Region-wise Pollution Levels
    # ---------------------------------------------------------
    elif viz_type == "Region-wise Pollution Levels":
        st.subheader("🌍 Region-wise Pollution Levels")

        if "REGION (DISPLAY)" not in data.columns:
            st.error("Region column missing in dataset")
        else:
            region_avg = data.groupby("REGION (DISPLAY)")["Value"].mean()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(region_avg.index, region_avg.values)
            ax.set_ylabel("Average Value")
            ax.set_title("Average Pollution Levels by Region")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

    # ---------------------------------------------------------
    # 📉 4. Pollution Range View (Low–High Interval)
    # ---------------------------------------------------------
    elif viz_type == "Pollution Range View":
        st.subheader("📉 Low–High Value Range")

        indicator_list = data["GHO (DISPLAY)"].dropna().unique()
        indicator = st.selectbox("Choose Indicator:", indicator_list)

        filtered = data[data["GHO (DISPLAY)"] == indicator]
        filtered = filtered.dropna(subset=["STARTYEAR", "Low", "High"])

        if filtered.empty:
            st.warning("No range data available for this indicator.")
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.fill_between(filtered["STARTYEAR"], filtered["Low"], filtered["High"], alpha=0.3)
            ax.plot(filtered["STARTYEAR"], filtered["Value"], color="black", marker="o")
            ax.set_xlabel("Year")
            ax.set_ylabel("Indicator Value")
            ax.set_title(f"Value Range Over Time: {indicator}")
            plt.tight_layout()
            st.pyplot(fig)
