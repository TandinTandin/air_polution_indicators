# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import requests
import json

st.set_page_config(page_title="Air Pollution Dashboard (HDX Data)", layout="wide")
st.title("🌍 Air Pollution Indicators Dashboard — HDX Data Source")

# --------------------------
# HDX Data Fetcher
# --------------------------
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def fetch_hdx_datasets(query="air pollution", limit=10):
    """Fetch datasets from HDX API"""
    try:
        url = "https://data.who.int/countries/064/api/3/action/package_search"
        params = {
            'q': query,
            'rows': limit
        }
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data['result']['results']
    except Exception as e:
        st.error(f"Failed to fetch from HDX API: {e}")
        return None

@st.cache_resource(ttl=3600)
def fetch_hdx_dataset_resources(package_id):
    """Fetch resources for a specific dataset"""
    try:
        url = f"https://data.humdata.org/api/3/action/package_show?id={package_id}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data['result']['resources']
    except Exception as e:
        st.error(f"Failed to fetch dataset resources: {e}")
        return None

@st.cache_resource(ttl=3600)
def load_data_from_url(url, file_type='csv'):
    """Load data from a URL"""
    try:
        if file_type == 'csv':
            return pd.read_csv(url)
        elif file_type in ['xlsx', 'xls']:
            return pd.read_excel(url)
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
    except Exception as e:
        st.error(f"Failed to load data from URL: {e}")
        return None

# --------------------------
# Data loader (HDX, upload, or local)
# --------------------------
data_source = st.sidebar.radio(
    "Choose data source:",
    ["HDX API", "Upload CSV", "Local File"]
)

data = None

if data_source == "HDX API":
    st.sidebar.header("HDX Data Search")
    
    # Search parameters
    search_query = st.sidebar.text_input("Search HDX datasets", "air pollution")
    search_limit = st.sidebar.slider("Max results", 5, 50, 10)
    
    if st.sidebar.button("Search HDX"):
        with st.spinner("Searching HDX datasets..."):
            datasets = fetch_hdx_datasets(search_query, search_limit)
            
        if datasets:
            st.sidebar.success(f"Found {len(datasets)} datasets")
            
            # Display dataset options
            dataset_options = {}
            for dataset in datasets:
                title = dataset.get('title', 'Untitled')
                org = dataset.get('organization', {}).get('title', 'Unknown')
                dataset_options[f"{title} ({org})"] = dataset
            
            selected_dataset_name = st.sidebar.selectbox(
                "Select dataset:",
                list(dataset_options.keys())
            )
            
            if selected_dataset_name:
                selected_dataset = dataset_options[selected_dataset_name]
                resources = fetch_hdx_dataset_resources(selected_dataset['id'])
                
                if resources:
                    # Filter for data resources (CSV, Excel)
                    data_resources = [
                        r for r in resources 
                        if r.get('format', '').lower() in ['csv', 'xlsx', 'xls']
                    ]
                    
                    if data_resources:
                        resource_options = {
                            f"{r.get('name', r.get('url', 'Unknown'))} ({r.get('format', 'unknown')})": r 
                            for r in data_resources
                        }
                        
                        selected_resource_name = st.sidebar.selectbox(
                            "Select data resource:",
                            list(resource_options.keys())
                        )
                        
                        if selected_resource_name and st.sidebar.button("Load Data"):
                            selected_resource = resource_options[selected_resource_name]
                            file_type = selected_resource.get('format', '').lower()
                            data_url = selected_resource['url']
                            
                            with st.spinner(f"Loading data from {data_url}..."):
                                data = load_data_from_url(data_url, file_type)
                                
                            if data is not None:
                                st.success(f"✅ Successfully loaded dataset: {selected_dataset['title']}")
                                st.info(f"📊 Shape: {data.shape} | 📈 Columns: {len(data.columns)}")
                    
                    else:
                        st.sidebar.warning("No CSV or Excel resources found in this dataset")
                
                # Display dataset info
                with st.expander("📋 Dataset Information"):
                    st.write(f"**Title:** {selected_dataset.get('title', 'N/A')}")
                    st.write(f"**Organization:** {selected_dataset.get('organization', {}).get('title', 'N/A')}")
                    st.write(f"**Description:** {selected_dataset.get('notes', 'No description available')}")
                    st.write(f"**Update Frequency:** {selected_dataset.get('data_update_frequency', 'N/A')}")
                    st.write(f"**Last Modified:** {selected_dataset.get('last_modified', 'N/A')}")
                    
                    if resources:
                        st.write("**Available Resources:**")
                        for resource in resources:
                            st.write(f"- {resource.get('name', 'Unnamed')} ({resource.get('format', 'unknown format')})")

elif data_source == "Upload CSV":
    st.sidebar.header("Upload Data")
    uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
            st.sidebar.success("✅ Uploaded CSV loaded successfully")
        except Exception as e:
            st.sidebar.error("❌ Failed to read uploaded CSV")
            st.sidebar.exception(e)

else:  # Local File
    st.sidebar.header("Local Data")
    local_path = "data/air_pollution_indicators_btn.csv"
    try:
        data = pd.read_csv(local_path)
        st.sidebar.success("✅ Local file loaded successfully")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Local file not found at {local_path}")

# Stop if no data loaded
if data is None:
    st.info("👆 Please select a data source and load data to continue")
    st.stop()

# --------------------------
# Dataset info
# --------------------------
st.sidebar.header("📊 Dataset Info")
st.sidebar.write(f"**Rows:** {data.shape[0]:,}")
st.sidebar.write(f"**Columns:** {data.shape[1]}")
st.sidebar.write(f"**Memory usage:** {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# --------------------------
# Navigation
# --------------------------
menu = st.sidebar.selectbox("Navigate", ["Overview", "Data Explorer", "Visualizations", "HDX Search"])

# --------------------------
# Overview
# --------------------------
if menu == "Overview":
    st.header("📋 Dataset Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Data Preview")
        st.dataframe(data.head(200), use_container_width=True)
    
    with col2:
        st.subheader("Basic Statistics")
        st.metric("Total Entries", f"{data.shape[0]:,}")
        st.metric("Total Features", f"{data.shape[1]}")
        st.metric("Missing Values", f"{data.isna().sum().sum():,}")
        st.metric("Duplicate Rows", f"{data.duplicated().sum():,}")
    
    st.subheader("Column Information")
    col_info = pd.DataFrame({
        "Data Type": data.dtypes.astype(str),
        "Missing Values": data.isna().sum(),
        "Missing %": (data.isna().sum() / len(data) * 100).round(2),
        "Unique Values": data.nunique(dropna=False)
    })
    st.dataframe(col_info, use_container_width=True)

# --------------------------
# Data Explorer
# --------------------------
elif menu == "Data Explorer":
    st.header("🔍 Data Explorer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Column Summary")
        selected_col = st.selectbox("Select column:", data.columns)
        if selected_col:
            col_data = data[selected_col]
            st.write(f"**Type:** {col_data.dtype}")
            st.write(f"**Non-null values:** {col_data.count():,}")
            st.write(f"**Null values:** {col_data.isna().sum():,}")
            st.write(f"**Unique values:** {col_data.nunique():,}")
            
            if pd.api.types.is_numeric_dtype(col_data):
                st.write(f"**Mean:** {col_data.mean():.2f}")
                st.write(f"**Std Dev:** {col_data.std():.2f}")
                st.write(f"**Min:** {col_data.min():.2f}")
                st.write(f"**Max:** {col_data.max():.2f}")
    
    with col2:
        st.subheader("Quick Stats")
        if pd.api.types.is_numeric_dtype(data[selected_col]):
            fig, ax = plt.subplots(figsize=(8, 4))
            data[selected_col].hist(bins=30, ax=ax, alpha=0.7)
            ax.set_title(f"Distribution of {selected_col}")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            value_counts = data[selected_col].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(8, 4))
            value_counts.plot(kind='bar', ax=ax, alpha=0.7)
            ax.set_title(f"Top 10 values in {selected_col}")
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

# --------------------------
# HDX Search Page
# --------------------------
elif menu == "HDX Search":
    st.header("🔎 HDX Dataset Search")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_query = st.text_input("Search term", "air quality pollution environmental")
        search_limit = st.slider("Number of results", 5, 30, 10)
    
    with col2:
        st.write("###")
        if st.button("Search HDX Catalog", type="primary"):
            with st.spinner("Searching HDX datasets..."):
                datasets = fetch_hdx_datasets(search_query, search_limit)
            
            if datasets:
                st.success(f"Found {len(datasets)} datasets")
                
                for i, dataset in enumerate(datasets, 1):
                    with st.expander(f"📁 {dataset.get('title', 'Untitled')}"):
                        col_a, col_b = st.columns([3, 1])
                        
                        with col_a:
                            st.write(f"**Organization:** {dataset.get('organization', {}).get('title', 'N/A')}")
                            st.write(f"**Description:** {dataset.get('notes', 'No description')[:200]}...")
                            st.write(f"**Last Modified:** {dataset.get('last_modified', 'N/A')}")
                        
                        with col_b:
                            if st.button(f"View Details", key=f"view_{i}"):
                                st.session_state.selected_dataset = dataset
                        
                        # Show resource formats
                        resources = dataset.get('resources', [])
                        if resources:
                            formats = list(set([r.get('format', 'unknown').upper() for r in resources]))
                            st.write(f"**Formats:** {', '.join(formats)}")
            else:
                st.error("No datasets found or API error")

# --------------------------
# Visualizations (robust)
# --------------------------
elif menu == "Visualizations":
    st.header("📈 Visualizations")
    
    # Controls
    sample_size = st.sidebar.number_input("Max rows to plot (0 = full)", min_value=0, value=10000, step=1000)
    use_sample = None if sample_size == 0 else int(sample_size)
    
    # Data preparation
    numeric = data.select_dtypes(include=[np.number]).copy()
    
    if numeric.shape[1] == 0:
        st.error("❌ No numeric columns detected. Visualizations require numeric data.")
        st.stop()
    
    # Clean numeric data
    before_cols = set(numeric.columns)
    numeric = numeric.dropna(axis=1, how="all")
    nunique = numeric.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        numeric = numeric.drop(columns=constant_cols)
    
    removed = list(before_cols - set(numeric.columns))
    if removed:
        st.info(f"ℹ️ Removed {len(removed)} invalid/constant/all-NaN columns: {removed}")
    
    if numeric.shape[1] < 1:
        st.error("❌ No valid numeric columns remain after cleaning.")
        st.stop()
    
    st.success(f"✅ Using {numeric.shape[1]} numeric columns for visualizations")
    
    # Helper function for sampling
    def get_df_for_plot(df):
        if use_sample is None or len(df) <= use_sample:
            return df
        else:
            return df.sample(use_sample, random_state=42)
    
    viz_type = st.selectbox(
        "Choose visualization type:",
        ["Correlation Heatmap", "Scatter Plot", "Line Graph", "Histogram", "Box Plot"]
    )
    
    # Correlation Heatmap
    if viz_type == "Correlation Heatmap":
        st.subheader("🔗 Correlation Heatmap")
        method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
        
        try:
            corr = get_df_for_plot(numeric).corr(method=method, numeric_only=True)
            if corr.shape[0] < 2:
                st.warning("Need at least 2 numeric columns for correlation.")
            else:
                fig, ax = plt.subplots(figsize=(12, 8))
                mask = np.triu(np.ones_like(corr, dtype=bool))  # Mask upper triangle
                sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", 
                           center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax)
                ax.set_title(f"Correlation Matrix ({method})", fontsize=14, pad=20)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Download option
                buf = BytesIO()
                corr.to_csv(buf)
                st.download_button(
                    "📥 Download correlation CSV", 
                    buf.getvalue(), 
                    "correlation_matrix.csv", 
                    "text/csv"
                )
        except Exception as e:
            st.error("Failed to compute or plot correlation.")
            st.exception(e)
    
    # Scatter Plot
    elif viz_type == "Scatter Plot":
        st.subheader("📊 Scatter Plot")
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X axis", numeric.columns, index=0)
        with col2:
            y_col = st.selectbox("Y axis", numeric.columns, index=min(1, len(numeric.columns)-1))
        
        jitter = st.checkbox("Add jitter (for discrete data)", value=False)
        size = st.slider("Marker size", 10, 200, 40)
        alpha = st.slider("Transparency", 0.1, 1.0, 0.7)
        
        try:
            df_plot = get_df_for_plot(numeric)
            df_clean = df_plot[[x_col, y_col]].dropna()
            
            if len(df_clean) < 2:
                st.warning("Not enough data points after removing NaN values.")
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                x = df_clean[x_col]
                y = df_clean[y_col]
                
                if jitter:
                    x_jitter = x + np.random.normal(0, x.std() * 0.01, len(x))
                    y_jitter = y + np.random.normal(0, y.std() * 0.01, len(y))
                    scatter = ax.scatter(x_jitter, y_jitter, s=size, alpha=alpha, cmap='viridis')
                else:
                    scatter = ax.scatter(x, y, s=size, alpha=alpha, cmap='viridis')
                
                # Add trend line
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2)
                
                ax.set_xlabel(x_col, fontsize=12)
                ax.set_ylabel(y_col, fontsize=12)
                ax.set_title(f"{y_col} vs {x_col}", fontsize=14)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Statistics
                correlation = df_clean[x_col].corr(df_clean[y_col])
                st.info(f"**Correlation:** {correlation:.3f}")
                
        except Exception as e:
            st.error("Failed to draw scatter plot")
            st.exception(e)
    
    # Line Graph
    elif viz_type == "Line Graph":
        st.subheader("📈 Line Graph")
        col = st.selectbox("Column to plot", numeric.columns)
        
        col1, col2 = st.columns(2)
        with col1:
            rolling = st.checkbox("Show rolling mean", value=False)
        with col2:
            window = st.slider("Rolling window", 2, 100, 7) if rolling else None
        
        try:
            df_plot = get_df_for_plot(numeric)
            series = df_plot[col].dropna().reset_index(drop=True)
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(series.index, series.values, label=col, alpha=0.7, linewidth=1)
            
            if rolling and window:
                rolling_mean = series.rolling(window=window, min_periods=1).mean()
                ax.plot(series.index, rolling_mean.values, linestyle="--", 
                       linewidth=2, label=f"Rolling mean (window={window})", color='red')
            
            ax.set_xlabel("Index")
            ax.set_ylabel(col)
            ax.set_title(f"Trend of {col}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        except Exception as e:
            st.error("Failed to draw line graph")
            st.exception(e)
    
    # Histogram
    elif viz_type == "Histogram":
        st.subheader("📊 Histogram")
        col = st.selectbox("Column", numeric.columns)
        
        col1, col2 = st.columns(2)
        with col1:
            bins = st.slider("Number of bins", 5, 100, 30)
        with col2:
            density = st.checkbox("Show density curve", value=True)
        
        try:
            arr = get_df_for_plot(numeric)[col].dropna()
            if arr.empty:
                st.warning("No non-null values for that column.")
            else:
                fig, ax = plt.subplots(figsize=(10, 5))
                n, bins, patches = ax.hist(arr, bins=bins, density=density, alpha=0.7, edgecolor='black')
                
                if density:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(arr)
                    x_vals = np.linspace(arr.min(), arr.max(), 100)
                    ax.plot(x_vals, kde(x_vals), 'r-', linewidth=2, label='Density')
                    ax.legend()
                
                ax.set_xlabel(col)
                ax.set_ylabel("Density" if density else "Count")
                ax.set_title(f"Distribution of {col}")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Statistics
                st.info(f"**Skewness:** {arr.skew():.3f} | **Kurtosis:** {arr.kurtosis():.3f}")
                
        except Exception as e:
            st.error("Failed to draw histogram")
            st.exception(e)
    
    # Box Plot
    elif viz_type == "Box Plot":
        st.subheader("📦 Box Plots")
        selected_cols = st.multiselect(
            "Select columns for box plots:",
            numeric.columns,
            default=numeric.columns[:min(5, len(numeric.columns))]
        )
        
        if selected_cols:
            try:
                df_plot = get_df_for_plot(numeric)[selected_cols]
                fig, ax = plt.subplots(figsize=(12, 6))
                df_plot.boxplot(ax=ax)
                ax.set_title("Box Plots of Selected Columns")
                ax.set_ylabel("Values")
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Outlier information
                st.subheader("Outlier Summary")
                for col in selected_cols:
                    Q1 = df_plot[col].quantile(0.25)
                    Q3 = df_plot[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df_plot[(df_plot[col] < lower_bound) | (df_plot[col] > upper_bound)][col]
                    st.write(f"**{col}:** {len(outliers)} outliers")
                    
            except Exception as e:
                st.error("Failed to draw box plots")
                st.exception(e)

st.caption("💡 Tip: Use the HDX Search to find and load air pollution datasets directly from the Humanitarian Data Exchange.")
