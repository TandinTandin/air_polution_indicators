Project Details: Air Pollution Indicators Analysis
1. Project Titlec

Air Pollution Indicators Analysis for Bhutan (WHO Data)

2. Project Overview

This project analyzes official World Health Organization (WHO) air-pollution indicators using a dataset containing 3,732 records. A Streamlit-based interactive dashboard was developed to visualize temporal trends, geographic patterns, and pollution-related health outcomes. The goal is to understand pollution exposure, population vulnerability, and national progress over time.

The dataset includes indicators related to:
- Ambient air pollution deaths
- Household air pollution (solid fuel use)
- Urban vs rural exposure
- -Annual mean PM2.5 levels
- Population proportions affected
- Confidence intervals (low–high values)
- The project allows users, policymakers, and researchers to explore how air pollution has evolved across years, regions, and demographic categories.


4. Objectives of the Project

1. Analyze trends in air pollution indicators across years.
2. Visualize pollution exposure across population groups (Urban vs Rural).
3. Identify potential health impacts using WHO mortality indicators.
4. Build an interactive Streamlit app for real-time data exploration.
5. Support policy insights by highlighting key pollution patterns.



5. Methodology
   
a. Data Processing
- Loaded and cleaned the CSV using pandas.
- Handled missing values.
- Standardized numerical columns (Value, Low, High).
- Extracted year for plotting.

b. Visualizations Developed
- Using Streamlit, Pandas, and Matplotlib, several visualizations were implemented:

1. Pollution values over time
- X-axis: STARTYEAR or YEAR (DISPLAY)
- Y-axis: Value
- Filters applied: Indicator, Dimension, Region

2. Indicator comparison
- Bar charts comparing indicators across years
- Scatter plots for correlation studies

3. Urban vs Rural exposure
- Grouped filtering based on DIMENSION (NAME)

4. Confidence interval visualization
- Error bars using Low and High columns

6. Key Features of the Streamlit Application



Sidebar Navigation Menu:
- Home
- Raw Data View
- Visualizations
- Indicator Insights
- Interactive Dropdowns:
- Select indicators
- Select year range
- Select category/dimension

Dynamic Charts:
- Line charts
- Bar charts
- Scatter plots

PM2.5 visualization
- Searchable Data Table to view the complete dataset.
