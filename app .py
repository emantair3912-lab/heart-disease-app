import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

model = pickle.load(open("svc_trained_model.pkl", "rb"))

SEX_LABELS = {1: "Male", 0: "Female"}
CP_LABELS = {
    1: "Typical Angina",
    2: "Atypical Angina",
    3: "Non-anginal Pain",
    4: "Asymptomatic",
}

st.title("Heart Disease Predictor")
st.write("SVC model trained on the UCI Heart Disease (Cleveland) dataset using 4 input attributes.")

age = st.number_input("Age", min_value=20, max_value=100, value=54)
sex = st.selectbox("Sex", options=list(SEX_LABELS), format_func=lambda k: SEX_LABELS[k])
chest_pain = st.selectbox("Chest pain type", options=list(CP_LABELS), format_func=lambda k: CP_LABELS[k], index=3)
max_hr = st.number_input("Maximum heart rate achieved", min_value=60, max_value=220, value=150)

if st.button("Predict"):
    row = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPain": [chest_pain],
        "MaxHeartRate": [max_hr],
    })
    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0][1]
    if pred == 1:
        st.error(f"Heart disease likely (probability {prob:.0%})")
    else:
        st.success(f"No heart disease likely (probability {prob:.0%})")
    st.caption("Educational project only, not medical advice.")
