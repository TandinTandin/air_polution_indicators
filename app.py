import streamlit as st
import pandas as pd
import altair as alt

# -----------------------------------------------------------
# 1. PAGE SETUP
# -----------------------------------------------------------
st.set_page_config(page_title="Air Pollution Dashboard", layout="wide")

st.title("Air Pollution Indicators Dashboard (Bhutan)")


# -----------------------------------------------------------
# 2. LOAD DATA
# -----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("air_pollution_indicators_btn (1).csv")
    return df

df = load_data()

st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Dataset", "Visualizations"])


# -----------------------------------------------------------
# 3. HOME SECTION
# -----------------------------------------------------------
if menu == "Home":
    st.subheader("Project Overview")
    st.write("""
        This dashboard visualizes air pollution indicators for Bhutan.
        You can explore the dataset, generate bar charts and line charts,
        and compare trends across years and indicators.
    """)


# -----------------------------------------------------------
# 4. DATASET SECTION
# -----------------------------------------------------------
elif menu == "Dataset":
    st.subheader("Dataset Preview")
    st.dataframe(df)


# -----------------------------------------------------------
# 5. VISUALIZATION SECTION (Bar + Line Charts Only)
# -----------------------------------------------------------
elif menu == "Visualizations":
    st.subheader("Visualizations")
    
    # Dropdowns
    indicator = st.selectbox("Select Indicator", df["Indicator Name"].unique())
    chart_type = st.selectbox("Chart Type", ["Bar Chart", "Line Chart"])

    # Filter data by indicator
    df_filtered = df[df["Indicator Name"] == indicator]

    # Ensure proper numeric sorting of years
    df_filtered = df_filtered.sort_values(by="Year")

    # BAR CHART
    if chart_type == "Bar Chart":
        st.write("### Bar Chart: Value Over Time")

        bar = (
            alt.Chart(df_filtered)
            .mark_bar(size=25)
            .encode(
                x=alt.X("Year:O", title="Year", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Value:Q", title="Value", scale=alt.Scale(nice=True)),
                tooltip=["Year", "Value"]
            )
            .properties(width=800, height=450)
        )
        st.altair_chart(bar, use_container_width=True)

    # LINE CHART
    else:
        st.write("### Line Chart: Value Over Time")

        line = (
            alt.Chart(df_filtered)
            .mark_line(point=True)
            .encode(
                x=alt.X("Year:O", title="Year", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Value:Q", title="Value", scale=alt.Scale(nice=True)),
                tooltip=["Year", "Value"]
            )
            .properties(width=800, height=450)
        )
        st.altair_chart(line, use_container_width=True)
