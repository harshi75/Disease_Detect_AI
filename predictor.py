import joblib
import pandas as pd
import numpy as np
import os

from symptom_aliases import symptom_aliases


# ------------------------------
# FILE PATHS
# ------------------------------

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_model.pkl"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "label_encoder.pkl"
)

SYMPTOM_PATH = os.path.join(
    BASE_DIR,
    "symptom_columns.pkl"
)


# ------------------------------
# LOAD MODEL FILES
# ------------------------------

model = joblib.load(MODEL_PATH)

encoder = joblib.load(ENCODER_PATH)

symptom_columns = joblib.load(SYMPTOM_PATH)


# ------------------------------
# PREDICTION FUNCTION
# ------------------------------

def predict_disease(user_symptoms):

    if not user_symptoms:
        return []


    # Normalize symptom names
    normalized_symptoms = []

    for symptom in user_symptoms:

        symptom = symptom.lower().strip()

        if symptom in symptom_aliases:
            symptom = symptom_aliases[symptom]

        normalized_symptoms.append(symptom)


    # Create binary symptom vector
    input_data = {}

    for symptom in symptom_columns:
        input_data[symptom] = 0


    for symptom in normalized_symptoms:

        if symptom in input_data:
            input_data[symptom] = 1


    # Convert into DataFrame
    input_df = pd.DataFrame([input_data])


    # Model prediction probabilities
    probabilities = model.predict_proba(input_df)[0]


    # Get top 5 predictions
    top_indices = np.argsort(probabilities)[-5:][::-1]


    results = []


    for index in top_indices:

        disease_name = encoder.inverse_transform(
            [index]
        )[0]


        confidence = round(
            probabilities[index] * 100,
            2
        )


        # Avoid unrealistic 100%
        confidence = min(confidence, 99.0)


        results.append(
            (
                disease_name,
                confidence
            )
        )


    return results
