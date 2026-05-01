import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load artifacts
model = joblib.load('best_model.pickle')
scaler = joblib.load('scalar.pickle')
le_gender = joblib.load('label_encoder_gender.pickle')
le_diabetic = joblib.load('label_encoder_diabetic.pickle')
le_smoker = joblib.load('label_encoder_smoker.pickle')

# Conversion Rate
USD_TO_INR = 40

st.set_page_config(page_title="HealthCare Expense Predictor", layout="wide")

st.title("🏥 Patient Bill & Insurance Estimator")
st.markdown("Enter patient clinical data and stay details to calculate the total estimated cost in ₹ (Rupees).")

with st.form("main_form"):
    st.subheader("1. Patient Demographics & Health")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", 18, 100, 30)
        gender = st.selectbox("Gender", le_gender.classes_)
        smoker = st.selectbox("Is the patient a smoker?", le_smoker.classes_)
        
    with col2:
        bmi = st.slider("BMI", 10.0, 60.0, 24.5, 0.1)
        diabetic = st.selectbox("Diabetic History", le_diabetic.classes_)
        bp = st.slider("Blood Pressure (Systolic)", 60, 200, 120)

    with col3:
        children = st.slider("Number of Dependents", 0, 5, 0)
        disease_type = st.selectbox("Primary Condition/Disease", 
                                   ["General Checkup", "Infectious Disease", "Cardiology", "Neurology", "Surgery"])
    
    st.markdown("---")
    st.subheader("2. Hospitalization Details")
    col_a, col_b = st.columns(2)
    
    with col_a:
        stay_days = st.number_input("Days of Stay", min_value=1, max_value=30, value=1)
    
    with col_b:
        checkups = st.multiselect("Checkups/Services Used", 
                                 ["X-Ray", "MRI/CT Scan", "Blood Test", "ECG", "Physiotherapy"],
                                 default=["Blood Test"])

    submitted = st.form_submit_button("Generate Estimated Bill (₹)")

if submitted:
    # 1. ML Prediction Logic (Base Insurance Cost)
    input_df = pd.DataFrame([[age, gender, bmi, bp, diabetic, children, smoker]], 
                            columns=['age', 'gender', 'bmi', 'bloodpressure', 'diabetic', 'children', 'smoker'])
    
    input_df['gender'] = le_gender.transform(input_df['gender'])
    input_df['diabetic'] = le_diabetic.transform(input_df['diabetic'])
    input_df['smoker'] = le_smoker.transform(input_df['smoker'])
    
    num_cols = ['age', 'bmi', 'bloodpressure', 'children']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    # Base prediction in USD, then convert to INR
    base_prediction_usd = model.predict(input_df)[0]
    base_prediction_inr = base_prediction_usd * USD_TO_INR
    
    # 2. Add-on Logic (Days and Services)
    # This keeps the app logical even though the original model didn't have these columns
    room_charge = stay_days * 2000  # ₹2000 per day
    checkup_costs = len(checkups) * 1500  # Avg ₹1500 per test
    disease_factor = {"General Checkup": 0, "Infectious Disease": 5000, "Cardiology": 15000, "Neurology": 12000, "Surgery": 30000}
    
    total_inr = base_prediction_inr + room_charge + checkup_costs + disease_factor[disease_type]

    # 3. Final Display
    st.success(f"### Total Estimated Expense: ₹{total_inr:,.2f}")
    
    with st.expander("View Bill Breakdown"):
        st.write(f"**Base Insurance Estimate (Converted):** ₹{base_prediction_inr:,.2f}")
        st.write(f"**Room Charges ({stay_days} days):** ₹{room_charge:,.2f}")
        st.write(f"**Checkup/Lab Fees:** ₹{checkup_costs:,.2f}")
        st.write(f"**Specialist Surcharge ({disease_type}):** ₹{disease_factor[disease_type]:,.2f}")
