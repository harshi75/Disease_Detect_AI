# 🩺 MedLens – AI-Powered Disease Prediction System

MedLens is an AI-powered healthcare application that predicts possible diseases based on user-selected symptoms using Machine Learning. The system analyzes symptoms and provides the Top-5 most probable diseases along with confidence scores, disease descriptions, precautions, and suggested treatments.

> ⚠️ This project is intended for educational and research purposes only and should not be used as a substitute for professional medical advice.

---

## 🚀 Features

✅ Predict diseases from symptoms

✅ Top-5 disease recommendations

✅ Confidence score for each prediction

✅ Disease descriptions

✅ Precautions and preventive measures

✅ Suggested treatments

✅ Interactive Streamlit web interface

✅ Supports 700+ diseases

✅ Machine Learning-based prediction engine

---

## 🏗️ System Architecture

```text
User Symptoms
       ↓
 Feature Encoding
       ↓
 Machine Learning Model
       ↓
 Disease Prediction
       ↓
 Top-5 Results
       ↓
 Description + Precautions + Treatment
```

---

## 🛠️ Tech Stack

### Frontend
- Streamlit

### Backend
- Python

### Machine Learning
- Scikit-Learn
- Naive Bayes
- Decision Tree
- Random Forest
- Ensemble Learning

### Data Processing
- Pandas
- NumPy

### Model Storage
- Joblib

---

## 📊 Model Performance

| Model | Accuracy |
|---------|---------|
| Decision Tree | 81.57% |
| Random Forest | 82.94% |
| Naive Bayes | **86.62%** |

### Best Results

- Top-1 Accuracy: **86.62%**
- Top-5 Accuracy: **98.14%**

---

## 📁 Project Structure

```text
MedLens/
│
├── app.py
├── predictor.py
├── train_model.py
├── disease_info.py
│
├── dataset.csv
│
├── model.pkl
├── label_encoder.pkl
│
├── requirements.txt
├── README.md
│
└── assets/
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/MedLens.git

cd MedLens
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Mac/Linux

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Train Model

```bash
python train_model.py
```

This generates:

```text
model.pkl
label_encoder.pkl
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## 📸 Application Workflow

1. Select symptoms.
2. Click **Predict Disease**.
3. View Top-5 predicted diseases.
4. Check confidence scores.
5. Read disease descriptions.
6. Review precautions and suggested treatments.

---

## 🎯 Use Cases

- Healthcare education
- Medical awareness
- Symptom-based disease screening
- AI/ML learning projects
- Research demonstrations
- Conference presentations

---

## 📈 Future Improvements

- Symptom similarity scoring
- Explainable AI (XAI)
- Disease category prediction
- RAG-based medical knowledge retrieval
- LLM-powered health assistant
- Doctor recommendation system
- PDF medical report generation
- Voice-based symptom input

---

## 👩‍💻 Author

**Harshita Singh**

B.Tech CSE (AI & ML)

Machine Learning | Healthcare AI | Data Science


---

## ⚠️ Disclaimer

MedLens is an educational and research-oriented machi
