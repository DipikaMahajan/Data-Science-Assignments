import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load("titanic_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Titanic Survival Prediction 🚢")

# User Inputs
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.radio("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=50.0)
embarked = st.selectbox("Embarked (C = Cherbourg, Q = Queenstown, S = Southampton)", ["C", "Q", "S"])

# Convert categorical inputs to numeric
sex = 1 if sex == "Female" else 0
embarked = {"C": 0, "Q": 1, "S": 2}[embarked]

# Prepare input for model
input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
input_data = scaler.transform(input_data)

# Make Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Survived! (Probability: {probability:.2f}) 🎉")
    else:
        st.error(f"Did not survive. (Probability: {probability:.2f}) 😢")
