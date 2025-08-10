import streamlit as st
import joblib
import numpy as np

scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

st.title("Customer Churn Prediction App")

st.divider()

st.write("Please enter the values and hit the Predict button to get the prediction.")

st.divider()

age = st.number_input("Enter age", min_value=10, max_value=100, value=30)
gender = st.selectbox("Enter gender", ['Male', 'Female'])
tenure = st.number_input("Enter tenure", min_value=0, max_value=130, value=10)
monthly_charges = st.number_input("Enter monthly charges", min_value=30.0, max_value=150.0)

st.divider()

predict_button = st.button("Predict")

st.divider()

if predict_button:
    gender_selected = 1 if gender == 'Female' else 0
    X = [age, gender_selected, tenure, monthly_charges]
    X_array = np.array(X)
    X_scaled = scaler.transform([X_array])
    prediction = model.predict(X_scaled)[0]
    predicted = 'Churn ' if prediction == 1 else 'Not Churn'

    st.balloons()
    st.write(f"The model predicts that the customer will {predicted} based on the provided inputs.")
else:
    st.write("Please fill in all the fields and click the Predict button to see the prediction.")
