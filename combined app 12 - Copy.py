# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 09:13:53 2025

@author: ayush
"""

import streamlit as st
import numpy as np
import pickle

# ---------------- Background ---------------- #
def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background: url('https://davidbaileyfurniture.co.uk/wp-content/uploads/2020/08/hospital-aesthetics-830-768x511.jpg');
            background-size: cover;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

# ---------------- Load Pre-trained Models ---------------- #
parkinsons_model = pickle.load(open(r"C:\\project expo 2\\parkinsons_model.pkl", "rb"))
diabetes_model = pickle.load(open(r"C:\\project expo 2\\C__diabetes.sav", "rb"))
heart_disease_model = pickle.load(open(r"C:\\project expo 2\\C__heart.sav", "rb"))

# ---------------- Sidebar Feature Descriptions ---------------- #
st.sidebar.title("Feature Guide")
feature_info = {
    "Diabetes": {
        "Pregnancies": "Number of times the person has been pregnant [0 - 17]",
        "Glucose Level": "Plasma glucose concentration [0 - 200]",
        "Blood Pressure": "Diastolic blood pressure (mm Hg) [0 - 122]",
        "Skin Thickness": "Triceps skin fold thickness (mm) [0 - 99]",
        "Insulin": "2-Hour serum insulin (mu U/ml) [0 - 846]",
        "BMI": "Body mass index (weight in kg/(height in m)^2) [0 - 67.1]",
        "Diabetes Pedigree Function": "Family diabetes likelihood [0.078 - 2.42]",
        "Age": "Age of the person [21 - 81]"
    },
    "Heart Disease": {
        "age": "Age of the patient [29 - 77]",
        "sex": "Gender (1 = male; 0 = female)",
        "cp": "Chest pain type (0-3)",
        "trestbps": "Resting blood pressure (mm Hg) [94 - 200]",
        "chol": "Serum cholesterol in mg/dl [126 - 564]",
        "fbs": "Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)",
        "restecg": "Resting ECG results (0-2)",
        "thalach": "Maximum heart rate achieved [71 - 202]",
        "exang": "Exercise induced angina (1 = yes; 0 = no)",
        "oldpeak": "ST depression due to exercise [0 - 6.2]",
        "slope": "Slope of the peak exercise ST segment [0 - 2]",
        "ca": "Major vessels colored by fluoroscopy [0 - 4]",
        "thal": "Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)"
    },
    "Parkinsons": {
        "MDVP:Fhi(Hz)": "Max vocal fundamental frequency [75 - 285]",
        "MDVP:Flo(Hz)": "Min vocal fundamental frequency [50 - 210]",
        "MDVP:Jitter(%)": "Variation in pitch [0.001 - 0.02]",
        "MDVP:Jitter(Abs)": "Absolute pitch jitter [0.00001 - 0.0003]",
        "MDVP:RAP": "Relative average perturbation [0.0005 - 0.02]",
        "MDVP:PPQ": "Five-point pitch variation [0.0005 - 0.02]",
        "Jitter:DDP": "Pitch perturbation difference [0.002 - 0.06]",
        "MDVP:Shimmer": "Variation in amplitude [0.01 - 0.1]",
        "MDVP:Shimmer(dB)": "Amplitude variation (dB) [0.1 - 1.5]",
        "Shimmer:APQ3": "3-point amplitude perturbation [0.005 - 0.05]",
        "Shimmer:APQ5": "5-point amplitude perturbation [0.006 - 0.07]",
        "MDVP:APQ": "Average amplitude perturbation [0.007 - 0.09]",
        "Shimmer:DDA": "Amplitude difference between cycles [0.015 - 0.15]",
        "NHR": "Noise-to-harmonics ratio [0.01 - 0.3]",
        "HNR": "Harmonics-to-noise ratio [5 - 35]",
        "RPDE": "Signal predictability [0.3 - 0.7]",
        "DFA": "Fractal scaling component [0.5 - 0.9]",
        "spread1": "Nonlinear pitch deviation [-7 - -1]",
        "spread2": "Amplitude deviation [0.001 - 0.5]",
        "D2": "Signal complexity [1.5 - 3.5]",
        "PPE": "Pitch period entropy [0.05 - 0.6]"
    }
}

for disease, features in feature_info.items():
    st.sidebar.markdown(f"### {disease} Features")
    for key, desc in features.items():
        st.sidebar.write(f"**{key}**: {desc}")

# ---------------- Sidebar Disease Selection ---------------- #
option = st.sidebar.radio("Select a disease to predict:", ["Diabetes", "Heart Disease", "Parkinsons"])

st.title("🩺 Multiple Disease Prediction System")

# ---------------- Diabetes Prediction ---------------- #
if option == "Diabetes":
    st.subheader("Diabetes Prediction")
    Pregnancies = st.text_input("Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure")
    SkinThickness = st.text_input("Skin Thickness")
    Insulin = st.text_input("Insulin")
    BMI = st.text_input("BMI")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    Age = st.text_input("Age")

    if st.button("Predict Diabetes"):
        try:
            input_data = np.array([[float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
                                    float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]])
            prediction = diabetes_model.predict(input_data)
            if prediction[0] == 1:
                st.error("🔴 High Risk: Diabetes Detected. Please consult a doctor.")
            else:
                st.success("🟢 Low Risk: No Diabetes Detected.")
        except ValueError:
            st.error("Please enter valid numbers.")

# ---------------- Heart Disease Prediction ---------------- #
elif option == "Heart Disease":
    st.subheader("Heart Disease Prediction")
    age = st.text_input("Age")
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.text_input("Chest Pain Type (0-3)")
    trestbps = st.text_input("Resting Blood Pressure")
    chol = st.text_input("Cholesterol")
    fbs = st.selectbox("Fasting Blood Sugar > 120?", ["Yes", "No"])
    restecg = st.text_input("Resting ECG (0-2)")
    thalach = st.text_input("Max Heart Rate")
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.text_input("ST Depression (Old Peak)")
    slope = st.text_input("Slope")
    ca = st.text_input("Number of Major Vessels (0-4)")
    thal = st.text_input("Thalassemia (1-3)")

    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    if st.button("Predict Heart Disease"):
        try:
            input_data = np.array([[float(age), sex, int(cp), float(trestbps), float(chol), fbs, int(restecg),
                                    float(thalach), exang, float(oldpeak), int(slope), int(ca), int(thal)]])
            prediction = heart_disease_model.predict(input_data)
            if prediction[0] == 1:
                st.error("🔴 High Risk: Heart Disease Detected. Please consult a cardiologist.")
            else:
                st.success("🟢 Low Risk: No Heart Disease Detected.")
        except ValueError:
            st.error("Please enter valid numbers.")

# ---------------- Parkinson's Prediction ---------------- #
elif option == "Parkinsons":
    st.subheader("Parkinson's Disease Prediction")
    features = [
        "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ",
        "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ",
        "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]
    user_input = []
    for feature in features:
        val = st.text_input(feature)
        user_input.append(val)

    if st.button("Predict Parkinson's"):
        try:
            input_data = np.array([list(map(float, user_input))])
            prediction = parkinsons_model.predict(input_data)
            if prediction[0] == 1:
                st.error("🔴 High Risk: Parkinson's Detected.")
            else:
                st.success("🟢 Low Risk: No Parkinson's Detected.")
        except ValueError:
            st.error("Please enter valid numbers.")
