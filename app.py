import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dengue Risk Predictor",
    page_icon="🦟",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Inter:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

.main-header {
    text-align: center;
    padding: 2rem 0 1rem;
}

.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    background: linear-gradient(90deg, #f7971e, #ffd200);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.main-header p {
    color: #a0aec0;
    font-size: 1rem;
}

.card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 2rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1.5rem;
}

.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #ffd200;
    margin-bottom: 1rem;
    font-weight: 700;
}

.result-high {
    background: linear-gradient(135deg, rgba(229,62,62,0.25), rgba(197,48,48,0.15));
    border: 1px solid rgba(229, 62, 62, 0.5);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}

.result-low {
    background: linear-gradient(135deg, rgba(56,161,105,0.25), rgba(39,103,73,0.15));
    border: 1px solid rgba(56, 161, 105, 0.5);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}

.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
}

.result-sub {
    color: #cbd5e0;
    margin-top: 0.5rem;
    font-size: 0.95rem;
}

.disclaimer {
    color: #718096;
    font-size: 0.78rem;
    text-align: center;
    margin-top: 1rem;
}

/* Override Streamlit input label colors */
label {
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load model & scaler ────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("dengue_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🦟 Dengue Risk Predictor</h1>
    <p>Enter patient blood test results to assess dengue fever risk.</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ Could not load model files. Make sure `dengue_model.pkl` and `scaler.pkl` are in the same directory.\n\n`{load_error}`")
    st.stop()

# ── Input form ─────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Patient Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=25, step=1)
with col2:
    gender = st.selectbox("Gender", options=["Male", "Female", "Child"])

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Blood Test Results</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    hemoglobin   = st.number_input("Hemoglobin (g/dL)",              min_value=0.0, value=13.5,  step=0.1,  format="%.1f")
    wbc_count    = st.number_input("WBC Count (×10³/µL)",            min_value=0.0, value=6.0,   step=0.1,  format="%.1f")
    differential = st.number_input("Differential Count (%)",         min_value=0.0, value=60.0,  step=0.1,  format="%.1f")
with col4:
    rbc_count    = st.number_input("RBC Count (×10⁶/µL)",           min_value=0.0, value=4.5,   step=0.01, format="%.2f")
    platelet     = st.number_input("Platelet Count (×10³/µL)",       min_value=0.0, value=200.0, step=1.0,  format="%.1f")
    pdw          = st.number_input("Platelet Distribution Width (%)", min_value=0.0, value=12.0,  step=0.1,  format="%.1f")

st.markdown('</div>', unsafe_allow_html=True)

# ── Medians used during training (for missing-value imputation reference) ──────
MEDIANS = {
    "age":                       25.0,
    "hemoglobin_g_dl":           13.5,
    "wbc_count":                  6.0,
    "differential_count":        60.0,
    "rbc_count":                  4.5,
    "platelet_count":           200.0,
    "platelet_distribution_width": 12.0,
}

# IQR caps used during training
IQR_CAPS = {
    "platelet_count":             {"lower": 50.0,  "upper": 400.0},
    "platelet_distribution_width":{"lower":  8.0,  "upper":  20.0},
    "hemoglobin_g_dl":            {"lower":  8.0,  "upper":  18.0},
}

# Features that were log-transformed
LOG_FEATURES = ["platelet_count", "wbc_count", "platelet_distribution_width"]

# Features that were standard-scaled (order matters — same as scaler was fit on)
SCALE_FEATURES = ["age", "hemoglobin_g_dl", "wbc_count", "platelet_count", "platelet_distribution_width"]

# Final feature order expected by the model
FEATURE_ORDER = [
    "age", "hemoglobin_g_dl", "wbc_count", "differential_count",
    "rbc_count", "platelet_count", "platelet_distribution_width",
    "gender_Female", "gender_Male",
]

def preprocess(age, hemoglobin, wbc_count, differential, rbc_count, platelet, pdw, gender):
    data = {
        "age":                        age,
        "hemoglobin_g_dl":            hemoglobin,
        "wbc_count":                  wbc_count,
        "differential_count":         differential,
        "rbc_count":                  rbc_count,
        "platelet_count":             platelet,
        "platelet_distribution_width": pdw,
    }

    # 1. Fill missing values with medians (inputs here are never NaN, but kept for parity)
    for col, median in MEDIANS.items():
        if data[col] is None or np.isnan(data[col]):
            data[col] = median

    # 2. One-hot encode gender (drop 'Child')
    data["gender_Female"] = 1.0 if gender == "Female" else 0.0
    data["gender_Male"]   = 1.0 if gender == "Male"   else 0.0

    # 3. IQR capping for outliers
    for col, caps in IQR_CAPS.items():
        data[col] = np.clip(data[col], caps["lower"], caps["upper"])

    # 4. Log transformation (np.log1p)
    for col in LOG_FEATURES:
        data[col] = np.log1p(data[col])

    # 5. Standard scaling — build a row with scale-feature values in fit order
    scale_row = np.array([[data[f] for f in SCALE_FEATURES]])
    scaled    = scaler.transform(scale_row)[0]
    for i, f in enumerate(SCALE_FEATURES):
        data[f] = scaled[i]

    # 6. Assemble final feature vector in model's expected order
    feature_vector = np.array([[data[f] for f in FEATURE_ORDER]])
    return feature_vector

# ── Predict button ─────────────────────────────────────────────────────────────
predict_btn = st.button("🔍 Predict Dengue Risk", use_container_width=True, type="primary")

if predict_btn:
    with st.spinner("Analyzing patient data…"):
        try:
            X = preprocess(age, hemoglobin, wbc_count, differential, rbc_count, platelet, pdw, gender)
            prediction = model.predict(X)[0]
            proba      = model.predict_proba(X)[0]
            confidence = proba[int(prediction)] * 100

            if prediction == 1:
                st.markdown(f"""
                <div class="result-high">
                    <div class="result-title" style="color:#fc8181;">⚠️ HIGH RISK</div>
                    <div class="result-sub">
                        This patient shows indicators consistent with <strong>Dengue Fever</strong>.<br>
                        Model confidence: <strong>{confidence:.1f}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-low">
                    <div class="result-title" style="color:#68d391;">✅ LOW RISK</div>
                    <div class="result-sub">
                        No strong dengue indicators detected for this patient.<br>
                        Model confidence: <strong>{confidence:.1f}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown('<div class="disclaimer">⚠️ This tool is for research and screening purposes only. It is not a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.</div>', unsafe_allow_html=True)
