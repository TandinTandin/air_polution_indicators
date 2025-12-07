# app.py
import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Air Pollution Dashboard (Bhutan)", layout="wide")
st.title("Air Pollution Indicators Dashboard (Bhutan)")
st.write("Focused Visualizations: Bar Chart & Line Chart — with color options")

# ---------------------------
# Load data (try local path first, else allow upload)
# ---------------------------
@st.cache_data
def load_from_path(path: str):
    return pd.read_csv(path)

df = None
default_path = "air_pollution_indicators_btn (1).csv"  # adjust filename if different

if Path(default_path).exists():
    try:
        df = load_from_path(default_path)
    except Exception as e:
        st.warning(f"Failed to load default file '{default_path}': {e}")

if df is None:
    st.info("Upload the air pollution CSV (or place it in app folder with name "
            "'air_pollution_indicators_btn (1).csv').")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)

if df is None:
    st.stop()  # no data to work with

# ---------------------------
# Normalize & prepare columns
# ---------------------------
# Show original columns for debugging if needed
# st.write("Columns:", df.columns.tolist())

# Expected columns based on your dataset: adjust if different
# Prefer STARTYEAR for x-axis; fall back to YEAR (DISPLAY) if needed.
if "STARTYEAR" in df.columns:
    df["Year"] = pd.to_numeric(df["STARTYEAR"], errors="coerce")
elif "YEAR (DISPLAY)" in df.columns:
    # try extracting numeric year from YEAR (DISPLAY)
    df["Year"] = pd.to_numeric(df["YEAR (DISPLAY)"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
else:
    # fallback: try any column that looks like year
    df["Year"] = pd.to_numeric(df.filter(regex=r"year", axis=1).iloc[:, 0], errors="coerce")

# Indicator column (use GHO (DISPLAY) if present)
if "GHO (DISPLAY)" in df.columns:
    df["Indicator"] = df["GHO (DISPLAY)"].astype(str)
elif "Indicator Name" in df.columns:
    df["Indicator"] = df["Indicator Name"].astype(str)
else:
    # fallback: first text column
    text_cols = df.select_dtypes(include=["object"]).columns
    df["Indicator"] = df[text_cols[0]].astype(str) if len(text_cols) > 0 else "Indicator"

# Numeric Value column
if "Value" in df.columns:
    df["ValueNum"] = pd.to_numeric(df["Value"], errors="coerce")
elif "Numeric" in df.columns:
    df["ValueNum"] = pd.to_numeric(df["Numeric"], errors="coerce")
else:
    # try to find any numeric column besides Year
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    df["ValueNum"] = df[numeric_cols[0]] if len(numeric_cols) > 0 else pd.NA

# Drop rows without year or value
df = df.dropna(subset=["Year", "ValueNum"])

# Convert Year to string for ordinal x-axis (keeps spacing even)
df["YearStr"] = df["Year"].astype(int).astype(str)

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Controls")
menu = st.sidebar.radio("Go to", ["Home", "Dataset", "Visualizations"])
color = st.sidebar.selectbox("Choose color", ["steelblue", "seagreen", "orange", "crimson", "purple", "gray"])
top_n = st.sidebar.slider("Number of indicators to show in bar chart (top by avg value)", 3, 30, 10)

# ---------------------------
# Home
# ---------------------------
if menu == "Home":
    st.subheader("About")
    st.write("""
        This dashboard uses WHO-style air pollution indicators for Bhutan.
        Under *Visualizations* you can choose an indicator and view either a bar chart
        (values by year) or a line chart (trend over time). Use the sidebar to pick colors.
    """)
    st.write("Dataset rows:", len(df))

# ---------------------------
# Dataset preview
# ---------------------------
elif menu == "Dataset":
    st.subheader("Dataset Preview (first 200 rows)")
    st.dataframe(df.head(200))

# ---------------------------
# Visualizations: Bar & Line only
# ---------------------------
elif menu == "Visualizations":
    st.subheader("Visualizations")

    # Selection of indicators
    indicator_list = df["Indicator"].dropna().unique()
    selected_indicator = st.selectbox("Select an indicator (or choose 'All' for bar comparison):",
                                      ["All"] + list(indicator_list))

    chart_type = st.selectbox("Chart Type", ["Bar Chart (value by year)", "Line Chart (trend)"])

    # If All + Bar Chart: show top-N indicators by average value
    if selected_indicator == "All" and chart_type.startswith("Bar"):
        st.write(f"### Top {top_n} Indicators by Average Value")
        avg_vals = (df.groupby("Indicator")["ValueNum"]
                      .mean()
                      .dropna()
                      .sort_values(ascending=False)
                      .head(top_n)
                      .reset_index())
        avg_vals["IndicatorShort"] = avg_vals["Indicator"].str.slice(0, 120)  # avoid extremely long labels

        chart = (
            alt.Chart(avg_vals)
            .mark_bar()
            .encode(
                y=alt.Y("IndicatorShort:N", sort='-x', title="Indicator"),
                x=alt.X("ValueNum:Q", title="Average Value"),
                tooltip=[alt.Tooltip("Indicator:N"), alt.Tooltip("ValueNum:Q", format=".3f")]
            )
            .properties(width=900, height=50 + 40 * len(avg_vals))
            .configure_axis(labelFontSize=12, titleFontSize=13)
            .configure_view(strokeWidth=0)
        )
        # apply color
        chart = chart.configure_mark(color=color)
        st.altair_chart(chart, use_container_width=True)

    else:
        # Filter to the chosen indicator
        df_sel = df[df["Indicator"] == selected_indicator] if selected_indicator != "All" else df.copy()
        if df_sel.empty:
            st.warning("No data available for that selection.")
        else:
            # Aggregate by year in case there are multiple rows per year
            agg = (df_sel.groupby("YearStr", as_index=False)["ValueNum"]
                   .mean()
                   .sort_values(by="YearStr"))

            if chart_type.startswith("Bar"):
                st.write(f"### Bar Chart — {selected_indicator}")
                bar = (
                    alt.Chart(agg)
                    .mark_bar()
                    .encode(
                        x=alt.X("YearStr:O", title="Year"),
                        y=alt.Y("ValueNum:Q", title="Value", scale=alt.Scale(nice=True)),
                        tooltip=[alt.Tooltip("YearStr:N", title="Year"), alt.Tooltip("ValueNum:Q", title="Value", format=".3f")]
                    )
                    .properties(width=900, height=450)
                )
                bar = bar.configure_mark(color=color)
                st.altair_chart(bar, use_container_width=True)

            else:
                st.write(f"### Line Chart — {selected_indicator}")
                line = (
                    alt.Chart(agg)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("YearStr:O", title="Year"),
                        y=alt.Y("ValueNum:Q", title="Value", scale=alt.Scale(nice=True)),
                        tooltip=[alt.Tooltip("YearStr:N", title="Year"), alt.Tooltip("ValueNum:Q", title="Value", format=".3f")]
                    )
                    .properties(width=900, height=450)
                )
                line = line.configure_mark(color=color)
                st.altair_chart(line, use_container_width=True)

    # show small data table for context
    with st.expander("Show data used for this chart"):
        st.dataframe(df_sel[["Indicator", "YearStr", "ValueNum"]].sort_values(["Indicator","YearStr"]).reset_index(drop=True))
