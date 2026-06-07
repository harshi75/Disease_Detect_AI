from disease_info import disease_info
import streamlit as st
import pandas as pd

from predictor import predict_disease

# -----------------------------------
# PAGE SETTINGS
# -----------------------------------

st.set_page_config(
    page_title="DiseaseDetectAI",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------------
# TITLE
# -----------------------------------

st.title("🩺 DiseaseDetectAI")
st.subheader("AI-Powered Disease Prediction System")

st.write(
    "Select your symptoms below and let the AI predict possible diseases."
)

# -----------------------------------
# LOAD DATA
# -----------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

df = load_data()

symptom_columns = df.drop("disease", axis=1).columns
symptom_list = sorted(symptom_columns.tolist())

# -----------------------------------
# SYMPTOM SELECTION
# -----------------------------------

selected_symptoms = st.multiselect(
    "Choose Symptoms",
    symptom_list
)

# -----------------------------------
# PREDICTION
# -----------------------------------

if st.button("Predict Disease"):

    if len(selected_symptoms) == 0:

        st.warning("Please select at least one symptom.")

    else:

        results = predict_disease(selected_symptoms)

        st.subheader("Top Possible Diseases")

        for disease, confidence in results:

            st.success(disease)

            st.progress(min(int(confidence), 100))

            st.write(f"Confidence: {confidence:.2f}%")

            info = disease_info.get(disease.lower())

            if info:

                st.markdown("### Description")
                st.write(info["description"])

                st.markdown("### Precautions")

                for precaution in info["precautions"]:
                    st.write("•", precaution)

                st.markdown("### Suggested Treatment")
                st.write(info["treatment"])

            else:

                st.info(
                    "Disease information not available yet."
                )

            st.divider()
