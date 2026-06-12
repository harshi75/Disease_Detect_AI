from disease_info import disease_info
import streamlit as st
from predictor import predict_disease
import joblib

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="DiseaseDetectAI",
    page_icon="🧬",
    layout="wide"
)

# Load symptoms
symptom_columns = joblib.load("symptom_columns.pkl")
symptom_list = sorted(symptom_columns)


# --------------------------------
# CUSTOM CSS
# --------------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #020617);
    color: white;
}

/* Main width */
.main {
    max-width: 900px;
    margin: auto;
}


/* Animated title */
.title {
    text-align:center;
    font-size:56px;
    font-weight:900;
    background:linear-gradient(90deg,#38bdf8,#a78bfa);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    animation: glow 2s ease-in-out infinite alternate;
}

@keyframes glow {
    from {
        filter: drop-shadow(0 0 6px #38bdf8);
    }
    to {
        filter: drop-shadow(0 0 22px #a78bfa);
    }
}


/* Subtitle */
.subtitle {
    text-align:center;
    font-size:18px;
    color:#cbd5e1;
    margin-bottom:25px;
}


/* Glass Cards */
.glass {
    background:rgba(255,255,255,0.08);
    border:1px solid rgba(255,255,255,0.15);
    border-radius:18px;
    padding:20px;
    backdrop-filter: blur(12px);
    box-shadow:0 10px 30px rgba(0,0,0,0.4);
    margin-bottom:20px;
}


/* Button */
.stButton button {
    background:linear-gradient(90deg,#38bdf8,#a78bfa);
    color:white;
    border:none;
    border-radius:12px;
    height:50px;
    font-size:18px;
    font-weight:700;
    transition:0.3s;
}


.stButton button:hover {
    transform:scale(1.02);
}


/* Selected symptom tags */
.stMultiSelect span {
    background:linear-gradient(90deg,#38bdf8,#a78bfa) !important;
    color:white !important;
    border-radius:8px !important;
}

</style>
""", unsafe_allow_html=True)



# --------------------------------
# HEADER
# --------------------------------

st.markdown(
    """
    <div class="title">
    🧬 DiseaseDetectAI
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="subtitle">
    AI-powered medical intelligence assistant
    </div>
    """,
    unsafe_allow_html=True
)


st.divider()


# --------------------------------
# FEATURES
# --------------------------------

st.markdown("""
<div class="glass">

<h3 style="color:#38bdf8;">
🚀 Why choose DiseaseDetectAI?
</h3>

✅ AI-powered disease prediction<br>
🧠 Machine Learning based diagnosis<br>
💊 Treatment & precaution suggestions<br>
⚡ Instant symptom analysis

</div>
""",
unsafe_allow_html=True)



# --------------------------------
# SYMPTOM INPUT
# --------------------------------

col1, col2, col3 = st.columns([1,2,1])


with col2:

    st.markdown("""
    <h2 style="color:white;">
    🔍 Select Your Symptoms
    </h2>

    <p style="color:#94a3b8;">
    Choose one or more symptoms for AI analysis
    </p>
    """,
    unsafe_allow_html=True)


    selected_symptoms = st.multiselect(
        "Choose Symptoms",
        symptom_list
    )


    predict_btn = st.button(
        "🚀 Analyze Symptoms",
        use_container_width=True
    )



# --------------------------------
# PREDICTION
# --------------------------------

if predict_btn:


    if not selected_symptoms:

        st.warning(
            "Please select at least one symptom."
        )


    else:


        with st.spinner(
            "🧠 AI is analyzing your symptoms..."
        ):

            results = predict_disease(
                selected_symptoms
            )


        col1, col2, col3 = st.columns(
            [1,2,1]
        )


        with col2:

            st.subheader(
                "🧠 AI Analysis Result"
            )


            for disease, confidence in results:


                st.markdown(
                    f"""
                    <div class="glass">

                    <h3 style="color:#38bdf8;">
                    🧬 {disease}
                    </h3>

                    <p style="
                    color:#22c55e;
                    font-size:18px;
                    font-weight:700;
                    ">
                    🔥 AI Confidence:
                    {confidence:.2f}%
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


                st.progress(
                    int(confidence)
                )


                info = disease_info.get(
                    disease.lower()
                )


                with st.expander(
                    "📘 View Medical Insights"
                ):


                    if info:

                        st.write(
                            "🧾 Description"
                        )

                        st.write(
                            info["description"]
                        )


                        st.write(
                            "⚠️ Precautions"
                        )

                        for precaution in info["precautions"]:

                            st.write(
                                "•",
                                precaution
                            )


                        st.write(
                            "💊 Treatment"
                        )

                        st.write(
                            info["treatment"]
                        )


                    else:

                        st.info(
                            "No medical data available for this condition."
                        )



# --------------------------------
# FOOTER
# --------------------------------

st.markdown("""
<br>
<hr>

<div style="
text-align:center;
color:#94a3b8;
font-size:14px;
">

🩺 DiseaseDetectAI <br>

Built with Machine Learning & Streamlit

<br><br>

⚠️ Educational purpose only.
Always consult a qualified healthcare professional
for medical advice.

</div>

""",
unsafe_allow_html=True)
