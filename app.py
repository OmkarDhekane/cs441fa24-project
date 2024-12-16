import streamlit as st
import pandas as pd
import numpy as np
import joblib


model = joblib.load("model.pkl")
scaler = joblib.load("scalar.pkl")

st.title("Heart Disease Prediction")

st.sidebar.header("Input Features")

if "user_data" not in st.session_state:
    st.session_state.user_data = pd.DataFrame(columns=["PatientId", "Age", "Sex", "Risk", "Probability"])
    st.session_state.user_id = 1  

def user_input_features():
    Age = st.sidebar.slider("Age", 20, 80, 50)
    Sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    RestingBP = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
    Cholesterol = st.sidebar.slider("Cholesterol Level", 100, 600, 200)
    FastingBS = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
    RestingECG = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    MaxHR = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 200, 150)
    ExerciseAngina = st.sidebar.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    Oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0, step=0.1)
    ChestPainType = st.sidebar.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
    ST_Slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

    data = {
        "Age": Age,
        "Sex": 1 if Sex == "Male" else 0,
        "ChestPainType": {"TA": 0, "ATA": 1, "NAP": 2, "ASY": 3}[ChestPainType],
        "RestingBP": RestingBP,
        "Cholesterol": Cholesterol,
        "FastingBS": 1 if FastingBS == "Yes" else 0,
        "RestingECG": {"Normal": 0, "ST": 1, "LVH": 2}[RestingECG],
        "MaxHR": MaxHR,
        "ExerciseAngina": 1 if ExerciseAngina == "Yes" else 0,
        "Oldpeak": Oldpeak,
        "ST_Slope": {"Up": 0, "Flat": 1, "Down": 2}[ST_Slope],
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

feature_order = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"
]

input_df = input_df[feature_order]

num_features = ["Age", "Cholesterol", "RestingBP", "MaxHR"]
scaled_features = input_df[num_features]
input_df[num_features] = scaler.transform(input_df[num_features])

def categorize_risk(probability):
    if probability > 0.7:
        return "High"
    elif probability > 0.4:
        return "Medium"
    else:
        return "Low"

if st.button("Predict"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    probability = prediction_proba[0][1]
    risk = categorize_risk(probability)

    original_values = scaler.inverse_transform(input_df[num_features])
    input_df[num_features] = original_values

    new_entry = {
        "PatientId": st.session_state.user_id,
        "Age": int(input_df.loc[0, "Age"]),
        "Sex": "Male" if input_df.loc[0, "Sex"] == 1 else "Female",
        "Risk": risk,
        "Probability": f"{probability * 100:.2f}%",
    }
    st.session_state.user_data = pd.concat([st.session_state.user_data, pd.DataFrame([new_entry])], ignore_index=True)

    st.session_state.user_id += 1

st.markdown(
    """
    <style>
    table {
        width: 100%;
        font-size: 1.2em;
        border-collapse: collapse;
    }
    th {
        text-align: right; 
        padding: 10px;
        border: 1px solid #ddd;
    }
    td {
        text-align: right;
        padding: 10px;
        border: 1px solid #ddd;
    }
    tr {
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if not st.session_state.user_data.empty:
    st.subheader("User Predictions Table")
    sorted_data = st.session_state.user_data.sort_values(by="Probability", ascending=False, key=lambda col: col.str.rstrip('%').astype(float))
    st.markdown(sorted_data.to_html(index=False, escape=False), unsafe_allow_html=True)

    csv_data = sorted_data.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv_data,
        file_name="heart_disease_predictions.csv",
        mime="text/csv"
    )