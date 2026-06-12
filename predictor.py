import streamlit as st
import joblib
import pandas as pd
import numpy as np
from symptom_aliases import symptom_aliases
@st.cache_resource
def load_model():

    model = joblib.load("best_model.pkl")
    encoder = joblib.load("label_encoder.pkl")

    return model, encoder

model, encoder = load_model()

# Load dataset
df = pd.read_csv("dataset.csv")

# Get symptom columns
symptom_columns = df.drop("disease", axis=1).columns


def predict_disease(user_symptoms):
    normalized_symptoms = []

    for symptom in user_symptoms:

        symptom = symptom.lower()

        if symptom in symptom_aliases:
            symptom = symptom_aliases[symptom]

            normalized_symptoms.append(symptom)

        user_symptoms = normalized_symptoms

    # Create symptom vector
        input_data = {}

        for symptom in symptom_columns:
            input_data[symptom] = 0

    # Mark selected symptoms
        for symptom in user_symptoms:

            symptom = symptom.lower().strip()

            if symptom in input_data:
                input_data[symptom] = 1

    # Convert to dataframe
        input_df = pd.DataFrame([input_data])

    # Get probabilities
        probabilities = model.predict_proba(input_df)[0]
    # Symptom similarity scoring
        similarity_scores = []

        for index, disease_name in enumerate(encoder.classes_):

            disease_rows = df[df["disease"] == disease_name]

            if len(disease_rows) == 0:
                similarity_scores.append(0)
                continue

            disease_symptoms = disease_rows.iloc[0, :-1]

            matched = 0

            for symptom in user_symptoms:
                if symptom in disease_symptoms.index:
                    if disease_symptoms[symptom] == 1:
                        matched += 1
            if len(user_symptoms) == 0:
                similarity = 0
            else:
                similarity = matched / len(user_symptoms)

            similarity_scores.append(similarity)

    # Top 3 predictions
    # Combine ML probability + similarity
        final_scores = (
            0.7 * probabilities +
            0.3 * np.array(similarity_scores)
        )

        top_indices = np.argsort(final_scores)[-5:][::-1]

        results = []

        for index in top_indices:

            disease_name = encoder.inverse_transform([index])[0]

            confidence = final_scores[index] * 100
            confidence = round(confidence, 2)

            if confidence > 95:
                confidence = 95
        

            results.append((disease_name, confidence))

        return results
