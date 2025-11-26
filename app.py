import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Mental Health Stress Level Predictor")

# Load and prepare data
@st.cache_resource
def load_data():
    np.random.seed(42)
    size = 500
    
    data = pd.DataFrame({
        'age': np.random.randint(15, 60, size),
        'sleep_hours': np.random.uniform(4, 10, size),
        'social_interaction': np.random.randint(0, 7, size),
        'work_stress': np.random.randint(1, 10, size),
        'physical_activity': np.random.randint(0, 6, size),
        'mood_score': np.random.randint(1, 10, size)
    })
    
    # Simple stress calculation
    score = (data['work_stress'] * 0.5) + (10 - data['mood_score']) + (6 - data['physical_activity'])
    data['stress_level'] = pd.cut(score, bins=[0, 8, 14, 20], labels=['low', 'medium', 'high'])
    
    return data

# Train model
@st.cache_resource
def train_model(data):
    X = data.drop("stress_level", axis=1)
    y = data["stress_level"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

data = load_data()
model = train_model(data)

# Sidebar navigation
menu = st.sidebar.selectbox("Menu", ["Data Overview", "Charts", "Predict"])

if menu == "Data Overview":
    st.header("Dataset Preview")
    st.dataframe(data.head())
    st.header("Stress Level Distribution")
    st.bar_chart(data["stress_level"].value_counts())

elif menu == "Charts":
    st.header("Data Visualizations")
    
    chart_type = st.selectbox("Chart Type", ["Heatmap", "Histogram", "Scatter Plot"])
    
    if chart_type == "Heatmap":
        numeric_data = data.select_dtypes(include=['number'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
        st.pyplot()
    
    elif chart_type == "Histogram":
        col = st.selectbox("Select Column", data.columns[:-1])
        plt.hist(data[col], bins=20)
        st.pyplot()
    
    elif chart_type == "Scatter Plot":
        x_col = st.selectbox("X Axis", data.columns[:-1])
        y_col = st.selectbox("Y Axis", data.columns[:-1])
        plt.scatter(data[x_col], data[y_col])
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        st.pyplot()

elif menu == "Predict":
    st.header("Predict Stress Level")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider('Age', 15, 60, 25)
        sleep_hours = st.slider('Sleep Hours', 4.0, 10.0, 7.0)
        social = st.slider('Social Interaction', 0, 7, 3)
    
    with col2:
        work_stress = st.slider('Work Stress', 1, 10, 5)
        activity = st.slider('Physical Activity', 0, 6, 2)
        mood = st.slider('Mood Score', 1, 10, 6)

    if st.button("Predict Stress Level"):
        features = np.array([[age, sleep_hours, social, work_stress, activity, mood]])
        prediction = model.predict(features)[0]
        st.success(f"Predicted Stress Level: {prediction}")
