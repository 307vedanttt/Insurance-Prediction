import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
st.write("Files in current directory:", os.listdir("."))

# Load artifacts
model = joblib.load('best_model.pickle')
scaler = joblib.load('scalar.pickle')
le_gender = joblib.load('label_encoder_gender.pickle')
le_diabetic = joblib.load('label_encoder_diabetic.pickle')
le_smoker = joblib.load('label_encoder_smoker.pickle')

st.title("Health Insurance Payment Prediction App")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 0, 100, 30)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        children = st.number_input("Number of Children", 0, 8, 0)
        
    with col2:
        bp = st.number_input("Blood Pressure", 60, 200, 120)
        gender = st.selectbox("Gender", le_gender.classes_)
        diabetic = st.selectbox("Diabetic", le_diabetic.classes_)
        smoker = st.selectbox("Smoker", le_smoker.classes_)
        
    submitted = st.form_submit_button("Predict Payment")

if submitted:
    # Prepare input data
    input_df = pd.DataFrame([[age, gender, bmi, bp, diabetic, children, smoker]], 
                            columns=['age', 'gender', 'bmi', 'bloodpressure', 'diabetic', 'children', 'smoker'])
    
    # Encode & Scale
    input_df['gender'] = le_gender.transform(input_df['gender'])
    input_df['diabetic'] = le_diabetic.transform(input_df['diabetic'])
    input_df['smoker'] = le_smoker.transform(input_df['smoker'])
    
    num_cols = ['age', 'bmi', 'bloodpressure', 'children']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    # Predict
    prediction = model.predict(input_df)
    st.success(f"Estimated Insurance Payment: ${prediction[0]:,.2f}")
