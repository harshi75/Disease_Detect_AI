from disease_info import disease_info
import streamlit as st
from predictor import predict_disease
import joblib

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="DiseaseDetectAI",
    page_icon="🩺",
    layout="wide"
)

symptom_columns = joblib.load("symptom_columns.pkl")

symptom_list = sorted(symptom_columns)

# -------------------------------
# MODERN UI THEME (GRADIENT + GLASS CARDS)
# -------------------------------
st.markdown("""
<style>

    /* Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a, #1e293b, #0b1220);
        color: white;
    }

    /* Center container */
    .main {
        max-width: 900px;
        margin: auto;
    }

    /* Title */
    .title {
        text-align: center;
        font-size: 46px;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 20px;
    }

    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #cbd5e1;
        margin-bottom: 25px;
    }

    /* Glass Card */
    .glass {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 18px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        margin-bottom: 15px;
    }

    /* Button improvement */
    .stButton > button {
        background: linear-gradient(90deg, #38bdf8, #a78bfa);
        color: white;
        border: none;
        padding: 10px;
        border-radius: 10px;
        font-weight: 600;
    }

    .stButton > button:hover {
        transform: scale(1.02);
        transition: 0.2s;
    }

</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown('<div class="title">🩺 DiseaseDetectAI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered interactive symptom checker</div>', unsafe_allow_html=True)

st.divider()

# -------------------------------
# CENTERED INPUT BOX
# -------------------------------
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # -----------------------------------
    # SYMPTOM SELECTION
    # -----------------------------------

    selected_symptoms = st.multiselect(
        "Choose Symptoms",
        symptom_list
    )

    st.markdown('</div>', unsafe_allow_html=True)



    

    st.subheader("🔍 Select Symptoms")

    
    predict_btn = st.button("Analyze Symptoms 🚀", use_container_width=True)
# -------------------------------
# INTERACTIVE RESULTS
# -------------------------------
if predict_btn:

    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        results = predict_disease(selected_symptoms)

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:

            st.subheader("🧠 AI Analysis Result")

            for disease, confidence in results:

                st.markdown(f"""
                <div class="glass">
                    <h3 style="color:#38bdf8;">🧬 {disease}</h3>
                    <p style="color:#cbd5e1;"><b>Confidence:</b> {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

                st.progress(int(confidence))

                info = disease_info.get(disease.lower())

                with st.expander("📘 View Medical Insights"):
                    if info:
                        st.write("🧾 Description")
                        st.write(info["description"])

                        st.write("⚠️ Precautions")
                        for p in info["precautions"]:
                            st.write("•", p)

                        st.write("💊 Treatment")
                        st.write(info["treatment"])
                    else:
                        st.info("No medical data available for this condition.")
