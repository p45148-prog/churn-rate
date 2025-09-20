import streamlit as st
import pandas as pd
import pickle
import joblib
import numpy as np

# Load the saved scaler and model
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Scaler or model file not found. Please make sure 'scaler.pkl' and 'logistic_regression_model.pkl' are in the same directory.")
    st.stop()

# Function to preprocess input data
def preprocess_data(credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary):
    # Create a DataFrame from the input data
    data = {'credit_score': [credit_score],
            'country': [country],
            'gender': [gender],
            'age': [age],
            'tenure': [tenure],
            'balance': [balance],
            'products_number': [products_number],
            'credit_card': [credit_card],
            'active_member': [active_member],
            'estimated_salary': [estimated_salary]}
    df_input = pd.DataFrame(data)

    # Apply one-hot encoding (handle unseen categories if necessary)
    # Get the columns that were in X_scaled during training (excluding country and gender dummies)
    original_cols = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary']
    dummy_cols = ['country_Germany', 'country_Spain', 'gender_Male']

    # Apply one-hot encoding to input data
    df_input = pd.get_dummies(df_input, columns=['country', 'gender'], drop_first=True)

    # Ensure all dummy columns that were present during training are present in the input data DataFrame.
    # If a category is not present in the input, add a column of zeros.
    for col in dummy_cols:
        if col not in df_input.columns:
            df_input[col] = 0

    # Reorder columns to match the training data columns (this is important for scaling and prediction)
    # Make sure the order matches the X_scaled DataFrame columns
    # To get the correct order, we need to know the column order of X_scaled
    # Since we don't have direct access to the X_scaled columns here,
    # a robust way is to save the column order during training
    # For this example, I'll assume the order based on the training process: original_cols + dummy_cols
    # In a real deployment, you would save the X_scaled.columns list.

    # Let's assume the order is original_cols followed by dummy_cols (alphabetically for dummies)
    expected_columns_order = original_cols + sorted(dummy_cols) # Assuming alphabetical order for dummies

    # Ensure the input dataframe has all expected columns and in the correct order
    # Add missing columns with default value 0 (e.g., if 'country' was always 'France' in input)
    for col in expected_columns_order:
        if col not in df_input.columns:
            df_input[col] = 0

    # Select and reorder the columns according to the expected order
    df_input_ordered = df_input[expected_columns_order]

    # Scale the numerical features
    # We need to identify which columns are numerical for scaling
    # Assuming the original_cols are the ones that were scaled
    df_input_ordered[original_cols] = scaler.transform(df_input_ordered[original_cols])


    return df_input_ordered

# Streamlit App Interface
st.title("Bank Customer Churn Prediction")

st.write("Enter customer details to predict churn.")

# Input fields for customer data
credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, value=650)
country = st.selectbox("Country", ['France', 'Spain', 'Germany'])
gender = st.selectbox("Gender", ['Female', 'Male'])
age = st.number_input("Age", min_value=18, max_value=120, value=38)
tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)
balance = st.number_input("Balance", min_value=0.0, value=70000.0)
products_number = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
credit_card = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
active_member = st.selectbox("Is Active Member", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=100000.0)

# Prediction button
if st.button("Predict Churn"):
    # Preprocess the input data
    processed_data = preprocess_data(credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary)

    # Make prediction
    prediction = model.predict(processed_data)
    prediction_proba = model.predict_proba(processed_data)[:, 1] # Probability of churn

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error(f"This customer is likely to churn. (Probability: {prediction_proba[0]:.2f})")
    else:
        st.success(f"This customer is unlikely to churn. (Probability: {prediction_proba[0]:.2f})")
