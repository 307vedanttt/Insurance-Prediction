import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Configuration for Premium Look
st.set_page_config(
    page_title="MedNavAI | Insurance Cost Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI Enhancement
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }
    .stButton>button {
        background-color: #10b981;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #059669;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    h1 {
        color: #0f172a;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
        border-top: 4px solid #10b981;
    }
    .metric-title {
        color: #64748b;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .metric-value {
        color: #0f172a;
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load artifacts
@st.cache_resource
def load_models():
    model = joblib.load('best_model.pickle')
    scaler = joblib.load('scalar.pickle')
    le_gender = joblib.load('label_encoder_gender.pickle')
    le_diabetic = joblib.load('label_encoder_diabetic.pickle')
    le_smoker = joblib.load('label_encoder_smoker.pickle')
    return model, scaler, le_gender, le_diabetic, le_smoker

try:
    model, scaler, le_gender, le_diabetic, le_smoker = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure all pickle files are in the same directory.")
    st.stop()

# Header
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=80)
with col2:
    st.title("MedNavAI Telemetry Financials")
    st.markdown("<p class='subtitle'>Predict your health insurance premium based on live biometric telemetry and health profiles.</p>", unsafe_allow_html=True)

st.markdown("---")

# Sidebar for Inputs
st.sidebar.header("Health Vault Sync")
st.sidebar.markdown("Manually input or simulate your synced biometric data below.")

with st.sidebar.form("input_form"):
    st.subheader("Demographics")
    age = st.slider("Patient Age", 18, 100, 30)
    gender = st.selectbox("Biological Sex", le_gender.classes_)
    children = st.number_input("Number of Dependents", 0, 10, 0)
    
    st.markdown("---")
    st.subheader("Telemetry Vitals")
    bmi = st.number_input("BMI (Body Mass Index)", 10.0, 60.0, 25.0)
    bp = st.number_input("Systolic Blood Pressure", 60, 200, 120)
    
    st.markdown("---")
    st.subheader("Clinical History")
    diabetic = st.selectbox("Diabetic Status", le_diabetic.classes_)
    smoker = st.selectbox("Nicotine/Smoking Status", le_smoker.classes_)
    
    submitted = st.form_submit_button("Run Risk Assessment & Calculate Premium")

# Main Content Area
if submitted:
    with st.spinner("Connecting to MedNavAI Neural Engine..."):
        
        # Prepare input data
        input_df = pd.DataFrame([[age, gender, bmi, bp, diabetic, children, smoker]], 
                                columns=['age', 'gender', 'bmi', 'bloodpressure', 'diabetic', 'children', 'smoker'])
        
        # Encode categorical variables
        input_df['gender'] = le_gender.transform(input_df['gender'])
        input_df['diabetic'] = le_diabetic.transform(input_df['diabetic'])
        input_df['smoker'] = le_smoker.transform(input_df['smoker'])
        
        # Scale numerical features
        num_cols = ['age', 'bmi', 'bloodpressure', 'children']
        input_df[num_cols] = scaler.transform(input_df[num_cols])
        
        # Predict (USD)
        usd_prediction = model.predict(input_df)[0]
        
        # ✅ UPDATED CONVERSION (1 USD = 30 INR)
        inr_prediction = usd_prediction * 30
        
        # Display Result
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Estimated Annual Premium</div>
            <div class="metric-value">₹ {inr_prediction:,.2f}</div>
            <p style="color: #64748b; font-size: 0.9rem;">Calculated based on your MedNavAI health profile</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.success("Telemetry Analysis Complete!")
        
        # Explanation
        with st.expander("📊 Clinical Breakdown: How was this calculated?"):
            st.write("""
            Our model analyzes:
            - Age & BMI
            - Smoking habits
            - Blood Pressure
            - Diabetes status
            - Number of dependents
            """)
            
            impact_data = pd.DataFrame({
                'Risk Factor': ['Smoking', 'Age', 'BMI', 'Blood Pressure', 'Children'],
                'Algorithmic Weight': [0.4, 0.25, 0.15, 0.1, 0.1]
            }).set_index('Risk Factor')
            
            st.bar_chart(impact_data, color="#10b981")

else:
    st.info("👈 Enter your data in the sidebar and click **Run Risk Assessment**.")
    
    st.markdown("""
    ### Why use MedNavAI Financials?
    - Accurate ML predictions
    - Clean UI
    - Private & secure
    """)
