import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Air Pollution Indicators Dashboard (Bhutan)")
st.write("### Focused Visualizations: Line Charts & Bar Charts")

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
@st.cache_resource
def load_data():
    file_path = "data/air_pollution_indicators_btn.csv"  # change if needed
    df = pd.read_csv(file_path)
    return df

data = load_data()

# Convert key fields to numeric
data["Value"] = pd.to_numeric(data["Value"], errors="coerce")
data["STARTYEAR"] = pd.to_numeric(data["STARTYEAR"], errors="coerce")

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
menu = st.sidebar.selectbox(
    "Navigate",
    ["Dataset Overview", "Line Chart", "Bar Chart"]
)

# ---------------------------------------------------------
# Dataset Overview
# ---------------------------------------------------------
if menu == "Dataset Overview":
    st.header("Dataset Overview")
    st.dataframe(data.head())

    st.write("### Summary Statistics")
    st.dataframe(data.describe(include="all"))

    st.write("### Column Information")
    st.write(data.dtypes)

# ---------------------------------------------------------
# LINE CHART
# ---------------------------------------------------------
elif menu == "Line Chart":
    st.header("📈 Line Chart: Pollution Trend Over Time")

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
# BAR CHART
# ---------------------------------------------------------
elif menu == "Bar Chart":
    st.header("📊 Bar Chart: Top 10 Indicators")

    indicator_avg = (
        data.groupby("GHO (DISPLAY)")["Value"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(indicator_avg.index, indicator_avg.values)
    ax.set_xlabel("Average Value")
    ax.set_title("Top 10 Indicators by Average Pollution Value")
    ax.invert_yaxis()  # largest on top
    plt.tight_layout()

    st.pyplot(fig)
