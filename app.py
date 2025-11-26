# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import requests
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="WHO Health Data Dashboard", layout="wide")
st.title("🌍 WHO Health Indicators Dashboard — China")

# --------------------------
# WHO API Data Fetcher
# --------------------------
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def fetch_who_indicators(country_code="CHN", limit=50):
    """Fetch available indicators from WHO API for China"""
    try:
        url = f"https://ghoapi.azureedge.net/api/Indicator"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Filter for relevant health indicators
        indicators = data.get('value', [])
        health_keywords = ['air', 'pollution', 'mortality', 'death', 'health', 'environment', 'quality', 'pm2.5', 'pm10']
        
        relevant_indicators = [
            ind for ind in indicators 
            if any(keyword in ind.get('IndicatorName', '').lower() for keyword in health_keywords)
        ][:limit]
        
        return relevant_indicators
    except Exception as e:
        st.error(f"Failed to fetch WHO indicators: {e}")
        return None

@st.cache_resource(ttl=3600)
def fetch_who_data(country_code="CHN", indicator_codes=None):
    """Fetch WHO data for specific country and indicators"""
    if indicator_codes is None:
        indicator_codes = ["AIR_1", "AIR_2", "AIR_3"]  # Default air quality indicators
    
    all_data = []
    
    for indicator_code in indicator_codes:
        try:
            url = f"https://ghoapi.azureedge.net/api/{indicator_code}"
            params = {'$filter': f"SpatialDim eq '{country_code}'"}
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('value'):
                df = pd.DataFrame(data['value'])
                df['IndicatorCode'] = indicator_code
                all_data.append(df)
                
        except Exception as e:
            st.warning(f"Could not fetch data for {indicator_code}: {e}")
            continue
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return None

@st.cache_resource(ttl=3600)
def fetch_who_country_data(country_code="CHN"):
    """Fetch comprehensive WHO data for China with multiple health indicators"""
    
    # Common WHO indicator codes for air pollution and health
    air_quality_indicators = [
        "AIR_1",    # Ambient air pollution
        "AIR_2",    # Household air pollution
        "AIR_3",    # Air pollution mean annual exposure
        "AIR_4",    # Air pollution in cities
        "SDG_3_9_1", # Mortality from air pollution
    ]
    
    health_indicators = [
        "MDG_0000000026",  # Under-five mortality
        "MDG_0000000026",  # Infant mortality
        "NCD_0002",        # Cardiovascular diseases
        "NCD_0003",        # Cancer
        "NCD_0004",        # Chronic respiratory diseases
    ]
    
    # Try to fetch air quality data first
    data = fetch_who_data(country_code, air_quality_indicators)
    
    if data is None or data.empty:
        # Fallback: create sample data for demonstration
        st.info("Using demonstration data - real WHO API might be temporarily unavailable")
        return create_demo_data()
    
    return data

def create_demo_data():
    """Create demonstration data when WHO API is unavailable"""
    years = list(range(2000, 2023))
    
    demo_data = []
    indicators = {
        'PM2.5_Exposure': {'trend': 'decreasing', 'base': 65},
        'PM10_Exposure': {'trend': 'decreasing', 'base': 120},
        'Air_Pollution_Mortality': {'trend': 'decreasing', 'base': 150},
        'Respiratory_Diseases': {'trend': 'increasing', 'base': 45},
        'Cardiovascular_Mortality': {'trend': 'stable', 'base': 280}
    }
    
    for year in years:
        for indicator, config in indicators.items():
            if config['trend'] == 'decreasing':
                value = config['base'] * (0.97 ** (year - 2000))
            elif config['trend'] == 'increasing':
                value = config['base'] * (1.02 ** (year - 2000))
            else:
                value = config['base'] + np.random.normal(0, 5)
            
            # Add some random variation
            value += np.random.normal(0, value * 0.1)
            
            demo_data.append({
                'IndicatorCode': indicator,
                'TimeDim': year,
                'NumericValue': max(0, value),
                'Value': str(round(value, 2)),
                'SpatialDim': 'CHN',
                'IndicatorName': indicator.replace('_', ' ')
            })
    
    return pd.DataFrame(demo_data)

# --------------------------
# Data Processing Functions
# --------------------------
def process_who_data(raw_data):
    """Process and clean WHO data"""
    if raw_data is None:
        return None
    
    df = raw_data.copy()
    
    # Convert numeric columns
    numeric_columns = ['NumericValue', 'TimeDim']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Extract year from date if available
    if 'TimeDim' in df.columns:
        df['Year'] = df['TimeDim']
    
    # Handle different value columns
    if 'Value' in df.columns:
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    
    # Clean indicator names
    if 'IndicatorName' in df.columns:
        df['IndicatorName'] = df['IndicatorName'].fillna('Unknown Indicator')
    
    return df

# --------------------------
# Data Loader
# --------------------------
st.sidebar.header("🌐 Data Source Configuration")

data_source = st.sidebar.radio(
    "Choose data source:",
    ["WHO API - China", "Upload Custom Data", "Sample Demo Data"]
)

data = None

if data_source == "WHO API - China":
    st.sidebar.subheader("WHO China Data")
    
    country_name = "China"
    country_code = "CHN"
    
    if st.sidebar.button("Fetch WHO Data", type="primary"):
        with st.spinner("Fetching latest WHO health data for China..."):
            raw_data = fetch_who_country_data(country_code)
            data = process_who_data(raw_data)
        
        if data is not None:
            st.sidebar.success(f"✅ WHO data loaded for {country_name}")
            st.sidebar.write(f"📊 Indicators: {data['IndicatorCode'].nunique()}")
            st.sidebar.write(f"📅 Time range: {int(data['Year'].min())}-{int(data['Year'].max())}")
        else:
            st.sidebar.error("❌ Failed to fetch WHO data")

elif data_source == "Upload Custom Data":
    st.sidebar.subheader("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ Custom data loaded successfully")
        except Exception as e:
            st.sidebar.error("❌ Failed to read uploaded file")
            st.sidebar.exception(e)

else:  # Sample Demo Data
    st.sidebar.subheader("Demo Data")
    if st.sidebar.button("Load Demonstration Data"):
        with st.spinner("Loading demonstration data..."):
            data = process_who_data(create_demo_data())
        st.sidebar.success("✅ Demonstration data loaded")

# Stop if no data loaded
if data is None:
    st.info("""
    ## 🌍 WHO Health Data Dashboard
    
    Welcome! This dashboard provides access to WHO health indicators for China, 
    focusing on air pollution and related health metrics.
    
    **To get started:**
    1. Select **"WHO API - China"** and click "Fetch WHO Data" for real data
    2. Or use **"Sample Demo Data"** for demonstration purposes
    3. Upload your own data with **"Upload Custom Data"**
    
    *Data source: [WHO Global Health Observatory](https://data.who.int/countries/064)*
    """)
    st.stop()

# --------------------------
# Dataset Info
# --------------------------
st.sidebar.header("📊 Dataset Info")
st.sidebar.write(f"**Records:** {data.shape[0]:,}")
st.sidebar.write(f"**Indicators:** {data['IndicatorCode'].nunique() if 'IndicatorCode' in data.columns else 'N/A'}")
st.sidebar.write(f"**Time span:** {int(data['Year'].min()) if 'Year' in data.columns else 'N/A'}-{int(data['Year'].max()) if 'Year' in data.columns else 'N/A'}")

# --------------------------
# Navigation
# --------------------------
menu = st.sidebar.selectbox(
    "Navigate", 
    ["Dashboard", "Indicator Analysis", "Temporal Trends", "Comparative Analysis", "Data Explorer"]
)

# --------------------------
# Dashboard Overview
# --------------------------
if menu == "Dashboard":
    st.header("🏠 WHO Health Data Dashboard - China")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_indicators = data['IndicatorCode'].nunique() if 'IndicatorCode' in data.columns else 'N/A'
        st.metric("Total Indicators", total_indicators)
    
    with col2:
        years_covered = f"{int(data['Year'].min())}-{int(data['Year'].max())}" if 'Year' in data.columns else 'N/A'
        st.metric("Time Coverage", years_covered)
    
    with col3:
        total_records = f"{data.shape[0]:,}"
        st.metric("Total Records", total_records)
    
    with col4:
        latest_year = int(data['Year'].max()) if 'Year' in data.columns else 'N/A'
        st.metric("Latest Data Year", latest_year)
    
    # Recent trends visualization
    st.subheader("📈 Recent Trends Overview")
    
    if 'IndicatorCode' in data.columns and 'Year' in data.columns and 'NumericValue' in data.columns:
        # Get top indicators by recent values
        recent_data = data[data['Year'] >= (data['Year'].max() - 5)]
        top_indicators = recent_data.groupby('IndicatorCode')['NumericValue'].mean().nlargest(6).index.tolist()
        
        if top_indicators:
            # Use matplotlib as fallback if Plotly has issues
            try:
                fig = make_subplots(
                    rows=2, cols=3,
                    subplot_titles=top_indicators,
                    vertical_spacing=0.1
                )
                
                for i, indicator in enumerate(top_indicators):
                    ind_data = data[data['IndicatorCode'] == indicator].sort_values('Year')
                    row = i // 3 + 1
                    col = i % 3 + 1
                    
                    fig.add_trace(
                        go.Scatter(x=ind_data['Year'], y=ind_data['NumericValue'], 
                                  mode='lines+markers', name=indicator),
                        row=row, col=col
                    )
                
                fig.update_layout(height=600, showlegend=False, title_text="Top Indicators Trend Analysis")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("Plotly chart failed, using matplotlib instead")
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()
                
                for i, indicator in enumerate(top_indicators):
                    if i < len(axes):
                        ind_data = data[data['IndicatorCode'] == indicator].sort_values('Year')
                        axes[i].plot(ind_data['Year'], ind_data['NumericValue'], marker='o')
                        axes[i].set_title(indicator)
                        axes[i].grid(True, alpha=0.3)
                        axes[i].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
    
    # Indicator distribution
    st.subheader("📊 Indicators Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'IndicatorCode' in data.columns:
            indicator_counts = data['IndicatorCode'].value_counts().head(10)
            try:
                fig = px.bar(
                    x=indicator_counts.values,
                    y=indicator_counts.index,
                    orientation='h',
                    title="Top 10 Indicators by Data Points",
                    labels={'x': 'Number of Records', 'y': 'Indicator Code'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("Plotly bar chart failed, using matplotlib")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(range(len(indicator_counts)), indicator_counts.values)
                ax.set_yticks(range(len(indicator_counts)))
                ax.set_yticklabels(indicator_counts.index)
                ax.set_xlabel('Number of Records')
                ax.set_title('Top 10 Indicators by Data Points')
                st.pyplot(fig)
    
    with col2:
        if 'Year' in data.columns:
            yearly_counts = data['Year'].value_counts().sort_index()
            try:
                fig = px.line(
                    x=yearly_counts.index,
                    y=yearly_counts.values,
                    title="Data Coverage Over Time",
                    labels={'x': 'Year', 'y': 'Number of Records'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("Plotly line chart failed, using matplotlib")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(yearly_counts.index, yearly_counts.values, marker='o')
                ax.set_xlabel('Year')
                ax.set_ylabel('Number of Records')
                ax.set_title('Data Coverage Over Time')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

# --------------------------
# Indicator Analysis
# --------------------------
elif menu == "Indicator Analysis":
    st.header("🔍 Indicator Analysis")
    
    if 'IndicatorCode' in data.columns:
        available_indicators = data['IndicatorCode'].unique()
        selected_indicator = st.selectbox("Select Indicator", available_indicators)
        
        if selected_indicator:
            indicator_data = data[data['IndicatorCode'] == selected_indicator].sort_values('Year')
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"Trend: {selected_indicator}")
                
                if len(indicator_data) > 1:
                    try:
                        fig = px.line(
                            indicator_data, 
                            x='Year', 
                            y='NumericValue',
                            title=f"{selected_indicator} Trend Over Time",
                            markers=True
                        )
                        fig.update_layout(
                            xaxis_title="Year",
                            yaxis_title="Value",
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning("Plotly line chart failed, using matplotlib")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(indicator_data['Year'], indicator_data['NumericValue'], marker='o', linewidth=2)
                        ax.set_xlabel('Year')
                        ax.set_ylabel('Value')
                        ax.set_title(f'{selected_indicator} Trend Over Time')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                else:
                    st.warning("Not enough data points for trend analysis")
            
            with col2:
                st.subheader("Statistics")
                
                if not indicator_data.empty:
                    latest_value = indicator_data.loc[indicator_data['Year'].idxmax(), 'NumericValue']
                    avg_value = indicator_data['NumericValue'].mean()
                    min_value = indicator_data['NumericValue'].min()
                    max_value = indicator_data['NumericValue'].max()
                    
                    st.metric("Latest Value", f"{latest_value:.2f}")
                    st.metric("Average", f"{avg_value:.2f}")
                    st.metric("Minimum", f"{min_value:.2f}")
                    st.metric("Maximum", f"{max_value:.2f}")
                    
                    # Yearly change if enough data
                    if len(indicator_data) >= 2:
                        recent_years = indicator_data.nlargest(2, 'Year')
                        if len(recent_years) == 2:
                            change = ((recent_years.iloc[0]['NumericValue'] - recent_years.iloc[1]['NumericValue']) / 
                                     recent_years.iloc[1]['NumericValue'] * 100)
                            st.metric("Yearly Change", f"{change:+.1f}%")
            
            # Distribution analysis
            st.subheader("Distribution Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    fig = px.histogram(
                        indicator_data,
                        x='NumericValue',
                        title=f"Distribution of {selected_indicator}",
                        nbins=20
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning("Plotly histogram failed, using matplotlib")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(indicator_data['NumericValue'], bins=20, alpha=0.7, edgecolor='black')
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'Distribution of {selected_indicator}')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
            
            with col2:
                try:
                    fig = px.box(
                        indicator_data,
                        y='NumericValue',
                        title=f"Box Plot - {selected_indicator}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning("Plotly box plot failed, using matplotlib")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.boxplot(indicator_data['NumericValue'].dropna())
                    ax.set_ylabel('Value')
                    ax.set_title(f'Box Plot - {selected_indicator}')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

# --------------------------
# Temporal Trends
# --------------------------
elif menu == "Temporal Trends":
    st.header("📅 Temporal Trends Analysis")
    
    if 'IndicatorCode' in data.columns and 'Year' in data.columns:
        selected_indicators = st.multiselect(
            "Select indicators to compare:",
            data['IndicatorCode'].unique(),
            default=data['IndicatorCode'].unique()[:3] if len(data['IndicatorCode'].unique()) >= 3 else data['IndicatorCode'].unique()
        )
        
        if selected_indicators:
            trend_data = data[data['IndicatorCode'].isin(selected_indicators)]
            
            # Normalize data for better comparison
            trend_data_normalized = trend_data.copy()
            for indicator in selected_indicators:
                mask = trend_data_normalized['IndicatorCode'] == indicator
                min_val = trend_data_normalized.loc[mask, 'NumericValue'].min()
                max_val = trend_data_normalized.loc[mask, 'NumericValue'].max()
                if max_val > min_val:
                    trend_data_normalized.loc[mask, 'NormalizedValue'] = (
                        (trend_data_normalized.loc[mask, 'NumericValue'] - min_val) / (max_val - min_val)
                    )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Values")
                try:
                    fig = px.line(
                        trend_data,
                        x='Year',
                        y='NumericValue',
                        color='IndicatorCode',
                        title="Multiple Indicators Trend Comparison"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning("Plotly multi-line chart failed, using matplotlib")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for indicator in selected_indicators:
                        ind_data = trend_data[trend_data['IndicatorCode'] == indicator].sort_values('Year')
                        ax.plot(ind_data['Year'], ind_data['NumericValue'], marker='o', label=indicator, linewidth=2)
                    ax.set_xlabel('Year')
                    ax.set_ylabel('Value')
                    ax.set_title('Multiple Indicators Trend Comparison')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
            
            with col2:
                st.subheader("Normalized Comparison (0-1 scale)")
                try:
                    fig = px.line(
                        trend_data_normalized,
                        x='Year',
                        y='NormalizedValue',
                        color='IndicatorCode',
                        title="Normalized Trends for Better Comparison"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning("Plotly normalized chart failed")
                    st.info("Normalized comparison not available with matplotlib fallback")
            
            # Change analysis
            st.subheader("Change Analysis")
            change_data = []
            
            for indicator in selected_indicators:
                ind_data = trend_data[trend_data['IndicatorCode'] == indicator].sort_values('Year')
                if len(ind_data) >= 2:
                    first_value = ind_data.iloc[0]['NumericValue']
                    last_value = ind_data.iloc[-1]['NumericValue']
                    change_pct = ((last_value - first_value) / first_value * 100) if first_value != 0 else 0
                    
                    change_data.append({
                        'Indicator': indicator,
                        'First Year': ind_data.iloc[0]['Year'],
                        'Last Year': ind_data.iloc[-1]['Year'],
                        'First Value': first_value,
                        'Last Value': last_value,
                        'Change %': change_pct
                    })
            
            if change_data:
                change_df = pd.DataFrame(change_data)
                st.dataframe(change_df.style.format({
                    'First Value': '{:.2f}',
                    'Last Value': '{:.2f}',
                    'Change %': '{:+.2f}%'
                }), use_container_width=True)

# --------------------------
# Data Explorer
# --------------------------
elif menu == "Data Explorer":
    st.header("🔍 Data Explorer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Raw Data Preview")
        st.dataframe(data.head(100), use_container_width=True)
    
    with col2:
        st.subheader("Dataset Information")
        
        info_data = {
            "Column Name": [],
            "Data Type": [],
            "Non-Null Count": [],
            "Null Count": [],
            "Unique Values": []
        }
        
        for col in data.columns:
            info_data["Column Name"].append(col)
            info_data["Data Type"].append(str(data[col].dtype))
            info_data["Non-Null Count"].append(data[col].count())
            info_data["Null Count"].append(data[col].isnull().sum())
            info_data["Unique Values"].append(data[col].nunique())
        
        info_df = pd.DataFrame(info_data)
        st.dataframe(info_df, use_container_width=True)
    
    # Data download
    st.subheader("Data Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = data.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Data as CSV",
            data=csv,
            file_name="who_china_health_data.csv",
            mime="text/csv"
        )
    
    with col2:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        summary = data[numeric_cols].describe()
        csv_summary = summary.to_csv()
        st.download_button(
            label="📊 Download Summary Statistics",
            data=csv_summary,
            file_name="who_data_summary.csv",
            mime="text/csv"
        )
    
    with col3:
        # Create a processed version
        processed = data.copy()
        if 'Year' in processed.columns and 'IndicatorCode' in processed.columns:
            processed = processed.pivot_table(
                index='Year',
                columns='IndicatorCode',
                values='NumericValue',
                aggfunc='mean'
            ).reset_index()
        
        csv_processed = processed.to_csv(index=False)
        st.download_button(
            label="🔄 Download Processed Data",
            data=csv_processed,
            file_name="who_data_processed.csv",
            mime="text/csv"
        )

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Data Source: <a href='https://data.who.int/countries/064'>WHO Global Health Observatory - China</a></p>
        <p>This dashboard provides visualization and analysis of WHO health indicators for China</p>
    </div>
    """,
    unsafe_allow_html=True
)
