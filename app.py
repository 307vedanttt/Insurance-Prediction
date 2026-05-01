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

USD_TO_INR = 40

st.set_page_config(page_title="HealthCare AI Hub", layout="wide", page_icon="🩺")

# Sidebar for Navigation / Project Info
st.sidebar.title("HealthCare AI Hub")
st.sidebar.info("This tool integrates with your Doctor Consultation Platform to provide financial transparency.")

st.title("🏥 Intelligent Billing & Insurance Estimator")
st.markdown("---")

with st.form("main_form"):
    st.subheader("📋 Patient Clinical Profile")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", 18, 100, 30)
        gender = st.selectbox("Gender", le_gender.classes_)
        smoker = st.selectbox("Smoker Status", le_smoker.classes_)
        
    with col2:
        bmi = st.slider("BMI", 10.0, 60.0, 24.5, 0.1)
        diabetic = st.selectbox("Diabetic History", le_diabetic.classes_)
        bp = st.slider("Blood Pressure", 60, 200, 120)

    with col3:
        children = st.slider("Number of Dependents", 0, 5, 0)
        # Unified specialty list matching your Next.js triage and doctor filters
        disease_type = st.selectbox("Specialty Required", 
                                   ["Cardiology", "Neurology", "Gastroenterology", "Orthopedics", "Oncology", "Radiology"])
    
    st.subheader("🏥 Hospitalization & Triage Details")
    col_a, col_b = st.columns(2)
    
    with col_a:
        stay_days = st.number_input("Estimated Hospital Stay (Days)", 1, 30, 1)
        urgency = st.select_slider("Triage Urgency Level", options=["Low", "Medium", "High", "Critical"])
    
    with col_b:
        services = st.multiselect("Ancillary Services", 
                                 ["Lab Tests", "Radiology (X-Ray/MRI)", "Pharmacy", "Emergency Care", "Physiotherapy"],
                                 default=["Lab Tests"])

    submitted = st.form_submit_button("Generate AI Financial Analysis")

if submitted:
    # 1. Base ML Prediction
    input_df = pd.DataFrame([[age, gender, bmi, bp, diabetic, children, smoker]], 
                            columns=['age', 'gender', 'bmi', 'bloodpressure', 'diabetic', 'children', 'smoker'])
    
    input_df['gender'] = le_gender.transform(input_df['gender'])
    input_df['diabetic'] = le_diabetic.transform(input_df['diabetic'])
    input_df['smoker'] = le_smoker.transform(input_df['smoker'])
    
    num_cols = ['age', 'bmi', 'bloodpressure', 'children']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    base_inr = model.predict(input_df)[0] * USD_TO_INR
    
    # 2. Logic to complement your Full Stack App
    room_charge = stay_days * 2500
    service_costs = len(services) * 2000
    
    # Specialty Surcharges (Customized to your Next.js filters)
    specialty_pricing = {
        "Cardiology": 15000, "Neurology": 12000, "Gastroenterology": 8000, 
        "Orthopedics": 10000, "Oncology": 25000, "Radiology": 5000
    }
    
    total_inr = base_inr + room_charge + service_costs + specialty_pricing[disease_type]

    # 3. Final Output Display
    st.markdown("---")
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        st.success(f"### Total Estimated Bill: ₹{total_inr:,.2f}")
        st.write(f"Based on your triage for **{disease_type}**, we recommend consulting a specialist on the Doctors Page.")
        
    with res_col2:
        # Integrated Call-to-Action back to your Next.js platform
        st.markdown(f"### Next Steps")
        if urgency in ["High", "Critical"]:
            st.error("🚨 Immediate Consultation Advised")
        
        # Link back to your local or deployed Next.js Doctors page
        st.link_button(f"Find {disease_type} Specialist", f"http://localhost:3000/doctors?specialty={disease_type}")

    with st.expander("Detailed Cost Breakdown"):
        st.write(f"**Insurance Risk-Adjusted Base:** ₹{base_inr:,.2f}")
        st.write(f"**Room & Stay Charges:** ₹{room_charge:,.2f}")
        st.write(f"**Specialty Fees ({disease_type}):** ₹{specialty_pricing[disease_type]:,.2f}")
