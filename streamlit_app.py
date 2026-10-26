from pathlib import Path

import joblib
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Diabetes Classification Demo",
    page_icon="📊",
)

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("Diabetes Classification Demo")
st.caption("Educational model output—not a medical diagnosis.")

if not MODEL_PATH.exists():
    st.error("model.pkl is missing. Run train_model.py first.")
    st.stop()

model = load_model()

with st.form("prediction_form"):
    left, right = st.columns(2)

    with left:
        pregnancies = st.number_input(
            "Pregnancies", min_value=0, value=2, step=1
        )
        glucose = st.number_input(
            "Glucose", min_value=0.0, value=120.0
        )
        blood_pressure = st.number_input(
            "Blood Pressure (mm Hg)", min_value=0.0, value=70.0
        )
        skin_thickness = st.number_input(
            "Skin Thickness (mm)", min_value=0.0, value=20.0
        )

    with right:
        insulin = st.number_input(
            "Insulin", min_value=0.0, value=79.0
        )
        bmi = st.number_input(
            "BMI", min_value=0.0, value=25.0, step=0.1
        )
        pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            value=0.5,
            step=0.01,
            format="%.3f",
        )
        age = st.number_input(
            "Age", min_value=1, value=30, step=1
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    # This order must match the training features.
    inputs = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        pedigree,
        age,
    ]])

    prediction = int(model.predict(inputs)[0])
    probabilities = model.predict_proba(inputs)[0]
    class_index = list(model.classes_).index(prediction)
    confidence = float(probabilities[class_index])

    label = "Diabetic" if prediction == 1 else "Not Diabetic"

    st.subheader("Model Output")
    st.write(f"Predicted class: **{label}**")
    st.metric("Neighbor Vote Share", f"{confidence:.0%}")
    st.caption(
        "The vote share describes agreement among the five "
        "nearest neighbors. It is not a clinical risk estimate."
    )