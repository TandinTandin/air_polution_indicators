import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Air Pollution Indicators Dashboard (Bhutan)")

@st.cache_resource
def load_data():
    file_path = "data/air_pollution_indicators_btn.csv"   # adjust path if needed
    df = pd.read_csv(file_path)
    return df

data = load_data()

menu = st.sidebar.selectbox(
    "Navigate",
    ["Dataset Overview", "Visualizations"]
)

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

    numeric_cols = ["Value"]

    if "STARTYEAR" in data.columns:
        data["STARTYEAR"] = pd.to_numeric(data["STARTYEAR"], errors="coerce")

    data["Value"] = pd.to_numeric(data["Value"], errors="coerce")

    if viz_type == "Pollution Trend Over Time":
        st.subheader("📈 Pollution Trend Over Time")

        indicator_list = data["GHO (DISPLAY)"].dropna().unique()
        selected_indicator = st.selectbox("Select Indicator:", indicator_list)

        filtered = data[data["GHO (DISPLAY)"] == selected_indicator]
        filtered = filtered.dropna(subset=["STARTYEAR", "Value"])

        if filtered.empty:
            st.error("No data available for this indicator.")
        else:
            plt.figure(figsize=(10, 5))
            plt.plot(filtered["STARTYEAR"], filtered["Value"], marker="o")
            plt.xlabel("Year")
            plt.ylabel("Value")
            plt.title(f"Trend Over Time: {selected_indicator}")
            st.pyplot()

    elif viz_type == "Pollution Type Comparison":
        st.subheader("📊 Indicator Comparison")

        indicator_avg = data.groupby("GHO (DISPLAY)")["Value"].mean().sort_values(ascending=False)

        plt.figure(figsize=(12, 6))
        plt.bar(indicator_avg.index[:10], indicator_avg.values[:10])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Average Value")
        plt.title("Top 10 Indicators by Average Value")
        st.pyplot()

    elif viz_type == "Region-wise Pollution Levels":
        st.subheader("🌍 Region-wise Pollution Levels")

        if "REGION (DISPLAY)" not in data.columns:
            st.error("Region column missing in dataset")
        else:
            region_data = data.groupby("REGION (DISPLAY)")["Value"].mean()

            plt.figure(figsize=(10, 5))
            plt.bar(region_data.index, region_data.values)
            plt.xticks(rotation=45)
            plt.ylabel("Average Value")
            plt.title("Average Pollution Levels by Region")
            st.pyplot()

    elif viz_type == "Pollution Range View":
        st.subheader("📉 Low–High Value Range")

        indicator_list = data["GHO (DISPLAY)"].dropna().unique()
        indicator = st.selectbox("Choose Indicator:", indicator_list)

        filtered = data[data["GHO (DISPLAY)"] == indicator]
        filtered = filtered.dropna(subset=["STARTYEAR", "Low", "High"])

        if filtered.empty:
            st.warning("No range data available for this indicator.")
        else:
            plt.figure(figsize=(10, 5))
            plt.fill_between(filtered["STARTYEAR"], filtered["Low"], filtered["High"], alpha=0.3)
            plt.plot(filtered["STARTYEAR"], filtered["Value"], color="black")
            plt.xlabel("Year")
            plt.ylabel("Value")
            plt.title(f"Value Range Over Time: {indicator}")
            st.pyplot()
