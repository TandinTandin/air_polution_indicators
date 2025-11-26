# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.set_page_config(page_title="Air Pollution Dashboard (Robust)", layout="wide")
st.title("Air Pollution Indicators Dashboard — Robust Visualizations")

# --------------------------
# Data loader (upload OR local file)
# --------------------------
@st.cache_resource
def load_local(path: str):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
        st.success("Uploaded CSV loaded.")
    except Exception as e:
        st.error("Failed to read uploaded CSV.")
        st.exception(e)
        st.stop()
else:
    local_path = "data/air_pollution_indicators_btn.csv"
    data = load_local(local_path)
    if data is None:
        st.warning(
            "No uploaded file and local dataset not found. "
            "Please upload your CSV or add the file at 'data/air_pollution_indicators_btn.csv' in the repo."
        )
        st.stop()

# quick dataset info
st.sidebar.header("Dataset info")
st.sidebar.write(f"Rows: {data.shape[0]}")
st.sidebar.write(f"Columns: {data.shape[1]}")

# --------------------------
# Sidebar navigation
# --------------------------
menu = st.sidebar.selectbox("Navigate", ["Overview", "Visualizations", "Debug"])

# --------------------------
# Overview
# --------------------------
if menu == "Overview":
    st.header("Dataset preview")
    st.dataframe(data.head(200), use_container_width=True)

    st.header("Column types & missing")
    col_info = pd.DataFrame({
        "dtype": data.dtypes.astype(str),
        "num_missing": data.isna().sum(),
        "num_unique": data.nunique(dropna=False)
    })
    st.dataframe(col_info, use_container_width=True)

# --------------------------
# Debug page to help troubleshoot
# --------------------------
elif menu == "Debug":
    st.header("Debug: Numeric columns detection")
    numeric = data.select_dtypes(include=[np.number])
    st.write("Numeric columns detected:", list(numeric.columns))
    st.write("Sample of numeric data (first 10 rows):")
    st.dataframe(numeric.head(10), use_container_width=True)
    st.write("Non-numeric columns (first 20):")
    st.dataframe(data.select_dtypes(exclude=[np.number]).head(20), use_container_width=True)

# --------------------------
# Visualization (robust)
# --------------------------
elif menu == "Visualizations":
    st.header("Visualizations")

    # Controls
    sample_size = st.sidebar.number_input("Max rows to plot (0 = full)", min_value=0, value=10000, step=100)
    use_sample = None if sample_size == 0 else int(sample_size)

    numeric = data.select_dtypes(include=[np.number]).copy()

    st.subheader("Detected numeric columns")
    if numeric.shape[1] == 0:
        st.error("No numeric columns detected. Visualizations require numeric data.")
        st.stop()
    st.write(list(numeric.columns))

    # Clean numeric: drop all-NaN columns and constant columns
    before_cols = set(numeric.columns)
    numeric = numeric.dropna(axis=1, how="all")
    nunique = numeric.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        numeric = numeric.drop(columns=constant_cols)
    removed = list(before_cols - set(numeric.columns))
    if removed:
        st.info(f"Removed {len(removed)} invalid/constant/all-NaN columns: {removed}")

    if numeric.shape[1] < 1:
        st.error("No valid numeric columns remain after cleaning.")
        st.stop()

    # small helper to slice sample
    def get_df_for_plot(df):
        if use_sample is None:
            return df
        else:
            return df.head(use_sample)

    viz_type = st.selectbox("Choose visualization", ["Correlation Heatmap", "Scatter Plot", "Line Graph", "Histogram"])

    # --------------------------
    # Correlation Heatmap
    # --------------------------
    if viz_type == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
        try:
            corr = get_df_for_plot(numeric).corr(method=method)
            if corr.shape[0] < 2:
                st.warning("Need at least 2 numeric columns for correlation.")
            else:
                fig, ax = plt.subplots(figsize=(10, 7))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.4, ax=ax)
                ax.set_title(f"Correlation ({method})", pad=12)
                plt.tight_layout()
                st.pyplot(fig)
                # download
                buf = BytesIO()
                corr.to_csv(buf)
                st.download_button("Download correlation CSV", buf.getvalue(), "correlation.csv", "text/csv")
        except Exception as e:
            st.error("Failed to compute or plot correlation.")
            st.exception(e)

    # --------------------------
    # Scatter Plot
    # --------------------------
    elif viz_type == "Scatter Plot":
        st.subheader("Scatter Plot")
        x_col = st.selectbox("X column", numeric.columns, index=0)
        y_col = st.selectbox("Y column", numeric.columns, index=1 if numeric.shape[1] > 1 else 0)
        jitter = st.checkbox("Add small jitter (helpful if many identical values)", value=False)
        size = st.slider("Marker size", 10, 200, 40)
        try:
            df_plot = get_df_for_plot(numeric)
            x = df_plot[x_col].dropna()
            y = df_plot[y_col].dropna()
            # align indexes
            df_xy = pd.concat([x, y], axis=1).dropna()
            fig, ax = plt.subplots(figsize=(10, 6))
            xs = df_xy[x_col].values
            ys = df_xy[y_col].values
            if jitter:
                xs = xs + np.random.normal(scale=(np.nanstd(xs) if np.nanstd(xs) > 0 else 0.0001) * 0.01, size=xs.shape)
                ys = ys + np.random.normal(scale=(np.nanstd(ys) if np.nanstd(ys) > 0 else 0.0001) * 0.01, size=ys.shape)
            ax.scatter(xs, ys, s=size, alpha=0.7)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{y_col} vs {x_col}")
            ax.grid(True)
            st.pyplot(fig)
        except Exception as e:
            st.error("Failed to draw scatter plot")
            st.exception(e)

    # --------------------------
    # Line Graph
    # --------------------------
    elif viz_type == "Line Graph":
        st.subheader("Line Graph")
        col = st.selectbox("Column to plot", numeric.columns)
        rolling = st.checkbox("Show rolling mean (window)", value=False)
        window = st.slider("Rolling window (rows)", 1, 100, 5) if rolling else None
        try:
            df_plot = get_df_for_plot(numeric)
            series = df_plot[col].dropna().reset_index(drop=True)
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(series.index, series.values, label=col)
            if rolling and window and window > 1:
                ax.plot(series.index, series.rolling(window=window, min_periods=1).mean().values, linestyle="--", label=f"rolling({window})")
            ax.set_xlabel("Index")
            ax.set_ylabel(col)
            ax.set_title(f"Trend of {col}")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        except Exception as e:
            st.error("Failed to draw line graph")
            st.exception(e)

    # --------------------------
    # Histogram
    # --------------------------
    elif viz_type == "Histogram":
        st.subheader("Histogram")
        col = st.selectbox("Column", numeric.columns)
        bins = st.slider("Number of bins", 5, 100, 30)
        density = st.checkbox("Show density / normalized histogram", value=False)
        try:
            arr = get_df_for_plot(numeric)[col].dropna()
            if arr.empty:
                st.warning("No non-null values for that column.")
            else:
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.hist(arr, bins=bins, density=density, alpha=0.8)
                ax.set_xlabel(col)
                ax.set_ylabel("Density" if density else "Count")
                ax.set_title(f"Distribution of {col}")
                ax.grid(True)
                st.pyplot(fig)
        except Exception as e:
            st.error("Failed to draw histogram")
            st.exception(e)

st.caption("If visualizations still look wrong, copy-paste any exception shown above into the chat.")
