import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. LOAD THE TRAINED MODELS ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('loan_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("Error: .pkl files not found. Make sure 'loan_model.pkl' and 'scaler.pkl' are in the same folder as this script.")
    st.stop()

# --- 2. UI & INPUTS ---
st.title("💰 Loan Default Predictor")
st.write("Enter the borrower's details below to predict the probability of default.")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    loan_amnt = st.number_input("Loan Amount ($)", min_value=1000, max_value=40000, value=10000)
    int_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=30.0, value=12.5)
    installment = st.number_input("Monthly Installment ($)", min_value=50, max_value=1500, value=300)
    annual_inc = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=60000)
    dti = st.number_input("Debt-to-Income Ratio (DTI)", min_value=0.0, max_value=50.0, value=15.0)

with col2:
    term = st.selectbox("Loan Term (Months)", options=[36, 60])
    grade = st.selectbox("Loan Grade", options=['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    home_ownership = st.selectbox("Home Ownership", options=['RENT', 'MORTGAGE', 'OWN', 'ANY'])
    emp_length = st.selectbox("Employment Length", options=['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'])
    verification_status = st.selectbox("Verification Status", options=['Verified', 'Source Verified', 'Not Verified'])

# Additional Hidden/Default inputs (simplified for demo)
# In a real app, you would ask for these or calculate them
revol_bal = 15000  # Default average
revol_util = 50.0  # Default average
total_acc = 20     # Default average
sub_grade = 0      # Simplified (since we are using Grade)

# --- 3. PREPROCESSING ---
# We must encode the inputs exactly how the model was trained.
# Since we didn't save the LabelEncoders, we manually map them here 
# based on standard alphabetical logic (A=0, B=1, etc.)

grade_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6}
home_map = {'ANY':0, 'MORTGAGE':1, 'OWN':2, 'RENT':3} 
ver_map = {'Not Verified':0, 'Source Verified':1, 'Verified':2}
emp_map = {
    '< 1 year':0, '1 year':1, '2 years':2, '3 years':3, '4 years':4, 
    '5 years':5, '6 years':6, '7 years':7, '8 years':8, '9 years':9, '10+ years':10
}

# Construct the input array
# ORDER MATTERS: It must match the order of 'cols_to_use' in your training script
# ['loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_length', 
# 'home_ownership', 'annual_inc', 'verification_status', 'dti', 'revol_bal', 'revol_util', 'total_acc']

input_data = pd.DataFrame({
    'loan_amnt': [loan_amnt],
    'term': [term],
    'int_rate': [int_rate],
    'installment': [installment],
    'grade': [grade_map[grade]],
    'sub_grade': [grade_map[grade] * 5], # Approximation for subgrade
    'emp_length': [emp_map[emp_length]],
    'home_ownership': [home_map[home_ownership]],
    'annual_inc': [annual_inc],
    'verification_status': [ver_map[verification_status]],
    'dti': [dti],
    'revol_bal': [revol_bal],
    'revol_util': [revol_util],
    'total_acc': [total_acc]
})

# --- 4. PREDICTION ---
if st.button("Predict Default Risk"):
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Predict Probability
    probability = model.predict_proba(input_scaled)[0][1] # Probability of Class 1 (Default)
    prediction = model.predict(input_scaled)[0]
    
    st.divider()
    
    # Display Result
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.metric(label="Default Probability", value=f"{probability:.2%}")
    
    with col_res2:
        if probability > 0.5: # Threshold
            st.error("Risk Assessment: **HIGH RISK** (Likely to Default)")
        else:
            st.success("Risk Assessment: **LOW RISK** (Likely to Pay)")

    # Explanation Chart (Feature Importance for Logistic Regression)
    # We multiply coefficient by input value to show contribution
    st.subheader("Why this prediction?")
    coeffs = model.coef_[0]
    # Simple bar chart of top contributing features (approximate visualization)
    feature_names = input_data.columns
    importance = pd.DataFrame({'Feature': feature_names, 'Weight': coeffs})
    importance = importance.sort_values(by='Weight', key=abs, ascending=False).head(5)
    
    st.bar_chart(importance.set_index('Feature'))