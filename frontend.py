import streamlit as st
import pandas as pd
import numpy as np
import pickle
from function_call import bmi, age_group, lifestyle_risk, city_tier

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Insurance Prediction")
st.markdown("Please Enter the following details")

# User input fields
age = st.number_input("Enter your age", min_value=1, max_value=50, value=10)
weight = st.number_input("Enter your Weight (Kg)", min_value=30, max_value=100, value=50)
height = st.number_input("Enter your height (m)", min_value=0.5, max_value=3.0, value=1.2)
income_lpa = st.number_input("Enter your Income (in LPA)", min_value=1.2, max_value=25.0, value=5.0)

smoker = st.selectbox("Enter your smoking status", options=[True, False])
city = st.selectbox("Select your city", options=["Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
    "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
    "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri","Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"])
occupation = st.selectbox("Select your occupation", options=['retired', 'freelancer', 'student', 'government_job',
    'business_owner', 'unemployed', 'private_job'])

# Prediction logic
if st.button("Predict my category"):
    bmi_value = weight / (height ** 2)
    user_data = pd.DataFrame({
        "bmi": [bmi_value],
        "age_group": [age_group(age)],
        "lifestyle_risk": [lifestyle_risk(smoker, bmi_value)],
        "city_tier": [city_tier(city)],
        "income_lpa": [income_lpa],
        "occupation": [occupation]
    })

    prediction = model.predict(user_data)
    st.success(f"Your insurance category: {prediction[0]}")
