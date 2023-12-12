import streamlit as st
import joblib
import pandas as pd
from transformers import *

model = joblib.load('model.joblib')

st.title("Diabetres Prediction App")
st.image("diabetes.webp", use_column_width="auto")

st.write("### Patient Data Form")

with st.form("my_form"):
    gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0)
    age = st.slider("Age", min_value=1, max_value=100, value=50)
    hypertension = int(st.checkbox("Hypertension", value=1))
    heart_disease = int(st.checkbox("Heart Disease", value=1))
    smoking_history = st.radio("Smoking History", ['never', 'No Info', 'current', 'former', 'ever', 'not current'],
                               index=2)
    bmi = st.slider("BMI", min_value=10, max_value=40, value=20)
    HbA1c_level = st.slider("HbA1c Level", min_value=4, max_value=12, value=6)
    blood_glucose_level = st.slider("Blood Glucose Level", min_value=50, max_value=300,
                                    value=100)

    submitted = st.form_submit_button("Predict!")

    if submitted:
        new_data = {
            "gender": [gender],
            "age": [age],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease],
            "smoking_history": [smoking_history],
            "bmi": [bmi],
            "HbA1c_level": [HbA1c_level],
            "blood_glucose_level": [blood_glucose_level]
        }

        prediction = model.predict(pd.DataFrame(new_data))[0]

        label = "Positive ⚠️" if prediction == 1 else "Negative ✅"
        st.write("Result is:")
        st.write(f"### {label}")
