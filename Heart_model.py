#!/usr/bin/env python
# coding: utf-8

# In[19]:


import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model


# In[21]:


# Load trained Keras model

model = load_model("heart_attack_model.h5")


# In[27]:


st.set_page_config(page_title="Heart Attack Risk Prediction", layout="centered")
st.title("💓 Heart Attack Risk Prediction")

st.markdown("Enter the patient's health and lifestyle details below:")

with st.form("health_form"):
    age = st.slider("Age", 20, 100, 50)
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    smoking = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: "Smoker" if x == 1 else "Non-Smoker")
    hypertension = st.selectbox("Hypertension", [0, 1])
    diabetes = st.selectbox("Diabetes", [0, 1])
    obesity = st.selectbox("Obesity", [0, 1])
    cholesterol = st.slider("Cholesterol Level", 100, 400, 200)
    pollution = st.slider("Air Pollution Exposure", 0, 100, 50)
    physical_activity = st.slider("Physical Activity (hrs/week)", 0, 20, 5)
    diet_score = st.slider("Diet Score (0–10)", 0, 10, 5)
    stress = st.slider("Stress Level (0–10)", 0, 10, 5)
    alcohol = st.selectbox("Alcohol Consumption", [0, 1])
    family_history = st.selectbox("Family History of CVD", [0, 1])
    healthcare_access = st.selectbox("Healthcare Access (0 = Poor, 1 = Good)", [0, 1])
    rural_urban = st.selectbox("Rural or Urban (0 = Rural, 1 = Urban)", [0, 1])
    region = st.selectbox("Region (e.g. 0–4)", [0, 1, 2, 3, 4])
    province = st.selectbox("Province (e.g. 0–9)", list(range(10)))
    hospital_availability = st.selectbox("Hospital Availability", [0, 1])
    tcm_use = st.selectbox("Traditional Chinese Medicine Use", [0, 1])
    employment = st.selectbox("Employment Status (0 = Unemployed, 1 = Employed)", [0, 1])
    income = st.slider("Income Level", 0, 100000, 50000)
    bp = st.slider("Blood Pressure", 80, 200, 120)
    ckd = st.selectbox("Chronic Kidney Disease", [0, 1])
    prev_heart_attack = st.selectbox("Previous Heart Attack", [0, 1])
    cvd_score = st.slider("CVD Risk Score (0–100)", 0, 100, 50)

    submit = st.form_submit_button("🔍 Predict")

if submit:
    input_data = np.array([[age, gender, smoking, hypertension, diabetes, obesity,
                            cholesterol, pollution, physical_activity, diet_score,
                            stress, alcohol, family_history, healthcare_access,
                            rural_urban, region, province, hospital_availability,
                            tcm_use, employment, income, bp, ckd, prev_heart_attack,
                            cvd_score]])

    prediction = model.predict(input_data)[0][0]

    st.subheader("🔎 Prediction Result")
    if prediction > 0.5:
        st.error(f"⚠️ High Risk of Heart Attack (Risk Score: {prediction:.2f})")
    else:
        st.success(f"✅ Low Risk of Heart Attack (Risk Score: {prediction:.2f})")


# In[ ]:




