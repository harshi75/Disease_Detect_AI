import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import top_k_accuracy_score

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# -----------------------------
# LOAD DATASET
# -----------------------------
# Remove rare diseases
df = pd.read_csv("dataset.csv")
disease_counts = df["disease"].value_counts()

valid_diseases = disease_counts[
    disease_counts >= 10
].index

df = df[
    df["disease"].isin(valid_diseases)
]

print("Remaining Diseases:", len(valid_diseases))
# Features (symptoms)
X = df.drop("disease", axis=1)

# Labels (disease names)
y = df["disease"]

# Convert labels into numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# -----------------------------
# MODELS
# -----------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ),
    "Naive Bayes": GaussianNB(),
    
}

best_model = None
best_accuracy = 0
best_model_name = ""

# -----------------------------
# TRAIN + EVALUATE
# -----------------------------
for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)
    # Normal accuracy
    accuracy = accuracy_score(y_test, predictions)

    print(f"\n{name} Accuracy: {accuracy*100:.2f}%")

# Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

# Top-5 Accuracy
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X_test)

        top5 = top_k_accuracy_score(
            y_test,
            probabilities,
            k=5,
            labels=range(len(encoder.classes_))
        )

        print(f"\nTop-5 Accuracy: {top5*100:.2f}%")

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    print(f"{name} Accuracy: {accuracy * 100:.2f}%")

    # Save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# -----------------------------
# SAVE BEST MODEL
# -----------------------------
joblib.dump(best_model, "best_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")
joblib.dump(X.columns.tolist(), "symptom_columns.pkl")

print("\n==============================")
print(f"Best Model: {best_model_name}")
print(f"Best Accuracy: {best_accuracy * 100:.2f}%")
print("Model saved successfully!")
print("==============================")
