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

# Page Config for a professional look
st.set_page_config(page_title="Insurance Predictor", layout="centered")

st.title("🏥 Health Insurance Payment Predictor")
st.markdown("Adjust the sliders and details below to estimate insurance costs.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        # Age Slider: 18 to 100
        age = st.slider("Age", min_value=18, max_value=100, value=30)
        
        # BMI Slider: 10 to 60 with 1 decimal point
        bmi = st.slider("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        
        # Children Slider: 0 to 5 (most common range)
        children = st.slider("Number of Children", min_value=0, max_value=5, value=0)
        
    with col2:
        # Blood Pressure Slider: 60 to 200
        bp = st.slider("Blood Pressure (Systolic)", min_value=60, max_value=200, value=120)
        
        gender = st.selectbox("Gender", le_gender.classes_)
        diabetic = st.selectbox("Diabetic Status", le_diabetic.classes_)
        smoker = st.selectbox("Smoker?", le_smoker.classes_)
        
    st.markdown("---")
    submitted = st.form_submit_button("💰 Calculate Estimated Premium")

if submitted:
    # Prepare input data
    input_df = pd.DataFrame([[age, gender, bmi, bp, diabetic, children, smoker]], 
                            columns=['age', 'gender', 'bmi', 'bloodpressure', 'diabetic', 'children', 'smoker'])
    
    # Encode categorical features
    input_df['gender'] = le_gender.transform(input_df['gender'])
    input_df['diabetic'] = le_diabetic.transform(input_df['diabetic'])
    input_df['smoker'] = le_smoker.transform(input_df['smoker'])
    
    # Scale numerical features
    num_cols = ['age', 'bmi', 'bloodpressure', 'children']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    # Predict
    prediction = model.predict(input_df)
    
    # Display Result with high-contrast formatting
    st.subheader("Results")
    st.success(f"The estimated annual insurance payment is: **${prediction[0]:,.2f}**")
    
    # Professional Insight
    if smoker == "yes":
        st.info("Note: Smoking status significantly impacts the premium calculation.")
