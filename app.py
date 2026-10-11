"""
🩺 Diabetes Prediction Web App
================================
A Streamlit application that uses a Random Forest model
(trained on clinical data) to predict diabetes risk.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background & font */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }

    /* Card-like containers */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1rem;
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.6rem;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .main-header p {
        color: #b0bec5;
        font-size: 1.1rem;
        margin-top: -0.5rem;
    }

    /* Section titles */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #90caf9;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Result cards */
    .result-card {
        text-align: center;
        padding: 2rem;
        border-radius: 20px;
        margin-top: 1rem;
    }
    .result-positive {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        box-shadow: 0 8px 32px rgba(255, 65, 108, 0.3);
    }
    .result-negative {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        box-shadow: 0 8px 32px rgba(56, 239, 125, 0.3);
    }
    .result-card h2 {
        color: #fff;
        font-size: 2rem;
        margin-bottom: 0.3rem;
    }
    .result-card p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
    }

    /* Probability bar */
    .prob-container {
        background: rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
    .prob-label {
        color: #e0e0e0;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }
    .prob-bar-bg {
        background: rgba(255,255,255,0.1);
        border-radius: 8px;
        height: 28px;
        overflow: hidden;
        margin-bottom: 0.6rem;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 10px;
        font-weight: 700;
        font-size: 0.85rem;
        color: #fff;
        transition: width 0.8s ease;
    }

    /* Metric cards in sidebar */
    .metric-card {
        background: rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        text-align: center;
    }
    .metric-card .value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #64ffda;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #90a4ae;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff, #3a7bd5) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2rem !important;
        width: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.4) !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
    }

    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
        margin: 1.2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained model artifact from disk."""
    model_path = os.path.join(os.path.dirname(__file__), "diabetes_rf_model.joblib")
    artifact = joblib.load(model_path)
    return artifact


try:
    artifact = load_model()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

model = artifact["model"]
label_encoder = artifact["label_encoder"]
features = artifact["features"]
numeric_features = artifact["numeric_features"]
categorical_features = artifact["categorical_features"]
categorical_values = artifact["categorical_values"]

# ── Feature metadata for UI ─────────────────────────────────────────────────
FEATURE_INFO = {
    "HbA1c": {
        "label": "HbA1c (%)",
        "help": "Glycated hemoglobin — average blood sugar over 2-3 months.",
        "min": 3.0, "max": 15.0, "default": 5.7, "step": 0.1,
        "section": "blood_sugar",
    },
    "Fasting_Blood_Glucose": {
        "label": "Fasting Blood Glucose (mg/dL)",
        "help": "Blood sugar after 8+ hours of fasting.",
        "min": 50.0, "max": 300.0, "default": 100.0, "step": 1.0,
        "section": "blood_sugar",
    },
    "Postprandial_Blood_Glucose": {
        "label": "Postprandial Blood Glucose (mg/dL)",
        "help": "Blood sugar 2 hours after eating.",
        "min": 50.0, "max": 350.0, "default": 140.0, "step": 1.0,
        "section": "blood_sugar",
    },
    "Random_Blood_Glucose": {
        "label": "Random Blood Glucose (mg/dL)",
        "help": "Blood sugar at any random time of day.",
        "min": 50.0, "max": 400.0, "default": 140.0, "step": 1.0,
        "section": "blood_sugar",
    },
    "OGTT": {
        "label": "OGTT (mg/dL)",
        "help": "Oral Glucose Tolerance Test result.",
        "min": 50.0, "max": 350.0, "default": 140.0, "step": 1.0,
        "section": "blood_sugar",
    },
    "Insulin_Levels": {
        "label": "Insulin Levels (μU/mL)",
        "help": "Fasting insulin level.",
        "min": 1.0, "max": 80.0, "default": 15.0, "step": 0.5,
        "section": "metabolic",
    },
    "HOMA_IR": {
        "label": "HOMA-IR",
        "help": "Homeostatic model assessment for insulin resistance.",
        "min": 0.5, "max": 15.0, "default": 2.0, "step": 0.1,
        "section": "metabolic",
    },
    "C_Peptide": {
        "label": "C-Peptide (ng/mL)",
        "help": "C-Peptide level — indicator of insulin production.",
        "min": 0.5, "max": 12.0, "default": 2.5, "step": 0.1,
        "section": "metabolic",
    },
    "Fructosamine_Levels": {
        "label": "Fructosamine (μmol/L)",
        "help": "Short-term blood sugar control indicator (2-3 weeks).",
        "min": 150.0, "max": 500.0, "default": 250.0, "step": 1.0,
        "section": "metabolic",
    },
    "BMI": {
        "label": "BMI (kg/m²)",
        "help": "Body Mass Index.",
        "min": 12.0, "max": 55.0, "default": 25.0, "step": 0.1,
        "section": "body",
    },
    "Waist_Circumference": {
        "label": "Waist Circumference (inches)",
        "help": "Waist measurement — indicator of abdominal fat.",
        "min": 20.0, "max": 60.0, "default": 36.0, "step": 0.5,
        "section": "body",
    },
    "Blood_Pressure_Systolic": {
        "label": "Systolic BP (mmHg)",
        "help": "Systolic blood pressure (upper number).",
        "min": 80.0, "max": 200.0, "default": 120.0, "step": 1.0,
        "section": "cardiovascular",
    },
    "Blood_Pressure_Diastolic": {
        "label": "Diastolic BP (mmHg)",
        "help": "Diastolic blood pressure (lower number).",
        "min": 40.0, "max": 130.0, "default": 80.0, "step": 1.0,
        "section": "cardiovascular",
    },
    "Triglyceride_Levels": {
        "label": "Triglycerides (mg/dL)",
        "help": "Triglyceride blood levels.",
        "min": 30.0, "max": 400.0, "default": 150.0, "step": 1.0,
        "section": "cardiovascular",
    },
    "HDL_Cholesterol": {
        "label": "HDL Cholesterol (mg/dL)",
        "help": "High-density lipoprotein ('good' cholesterol).",
        "min": 15.0, "max": 100.0, "default": 50.0, "step": 1.0,
        "section": "cardiovascular",
    },
    # Categorical features
    "Family_History_of_Diabetes": {
        "label": "Family History of Diabetes",
        "help": "Does your family have a history of diabetes?",
        "section": "risk_factors",
    },
    "Hypertension": {
        "label": "Hypertension",
        "help": "Have you been diagnosed with hypertension?",
        "section": "risk_factors",
    },
    "Physical_Activity": {
        "label": "Regular Physical Activity",
        "help": "Do you engage in regular physical activity?",
        "section": "risk_factors",
    },
    "Obesity": {
        "label": "Obesity",
        "help": "Have you been diagnosed with obesity?",
        "section": "risk_factors",
    },
}

SECTIONS = {
    "blood_sugar": {"icon": "🩸", "title": "Blood Sugar Markers"},
    "metabolic":   {"icon": "⚗️", "title": "Metabolic Indicators"},
    "body":        {"icon": "📏", "title": "Body Measurements"},
    "cardiovascular": {"icon": "❤️", "title": "Cardiovascular Markers"},
    "risk_factors": {"icon": "⚠️", "title": "Risk Factors"},
}


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Model Information")

    st.markdown(f"""
    <div class="metric-card">
        <div class="value">Random Forest</div>
        <div class="label">Algorithm</div>
    </div>
    """, unsafe_allow_html=True)

    cv_acc = artifact.get("cv_accuracy", 0)
    test_acc = artifact.get("test_accuracy", 0)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{cv_acc * 100:.1f}%</div>
            <div class="label">CV Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{test_acc * 100:.1f}%</div>
            <div class="label">Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 📋 Features Used")
    st.markdown(f"**{len(numeric_features)}** numeric + **{len(categorical_features)}** categorical = **{len(features)}** total")

    with st.expander("View all features"):
        for f in features:
            info = FEATURE_INFO.get(f, {})
            st.markdown(f"- {info.get('label', f)}")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("### ℹ️ About")
    st.markdown("""
    This app uses a **Random Forest classifier** trained on clinical
    diabetes data. The model is embedded in a scikit-learn Pipeline
    that handles preprocessing automatically.

    > **Disclaimer**: This tool is for educational/demonstration
    > purposes only and should **not** be used for medical diagnosis.
    > Always consult a healthcare professional.
    """)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🩺 Diabetes Risk Predictor</h1>
    <p>Enter your clinical parameters below to assess diabetes risk using AI</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# ── Input Form ───────────────────────────────────────────────────────────────
input_data = {}

# Group features by section
section_features: dict[str, list[str]] = {}
for feat, info in FEATURE_INFO.items():
    sec = info["section"]
    section_features.setdefault(sec, []).append(feat)

# Render each section
for sec_key, sec_meta in SECTIONS.items():
    feats = section_features.get(sec_key, [])
    if not feats:
        continue

    st.markdown(
        f'<div class="section-title">{sec_meta["icon"]} {sec_meta["title"]}</div>',
        unsafe_allow_html=True,
    )

    if sec_key == "risk_factors":
        # Render categorical features as toggle columns
        cols = st.columns(len(feats))
        for col, feat in zip(cols, feats):
            info = FEATURE_INFO[feat]
            with col:
                val = st.toggle(info["label"], value=False, help=info["help"])
                input_data[feat] = "Yes" if val else "No"
    else:
        # Render numeric features as number inputs in multi-column layout
        cols = st.columns(min(len(feats), 3))
        for i, feat in enumerate(feats):
            info = FEATURE_INFO[feat]
            with cols[i % len(cols)]:
                input_data[feat] = st.number_input(
                    info["label"],
                    min_value=info["min"],
                    max_value=info["max"],
                    value=info["default"],
                    step=info["step"],
                    help=info["help"],
                )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# ── Prediction ───────────────────────────────────────────────────────────────
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    predict_button = st.button("🔬  Analyze Diabetes Risk", use_container_width=True)

if predict_button:
    # Build input DataFrame in the exact feature order the model expects
    input_df = pd.DataFrame([input_data])[features]

    # Get prediction & probabilities
    proba = model.predict_proba(input_df)[0]
    pred_idx = int(np.argmax(proba))
    prediction = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx])

    probabilities = {
        cls: float(p) for cls, p in zip(label_encoder.classes_, proba)
    }

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Result card
    is_positive = prediction == "Positive"
    card_class = "result-positive" if is_positive else "result-negative"
    emoji = "⚠️" if is_positive else "✅"
    status_text = "High Risk Detected" if is_positive else "Low Risk"
    advice = (
        "The model indicates an elevated risk of diabetes. Please consult a healthcare professional for proper evaluation."
        if is_positive
        else "The model indicates a low risk of diabetes based on the provided parameters. Continue maintaining a healthy lifestyle!"
    )

    st.markdown(f"""
    <div class="result-card {card_class}">
        <h2>{emoji} {status_text}</h2>
        <p>Prediction: <strong>{prediction}</strong> &nbsp;|&nbsp; Confidence: <strong>{confidence * 100:.1f}%</strong></p>
        <p style="margin-top:0.8rem; font-size:0.95rem; opacity:0.9;">{advice}</p>
    </div>
    """, unsafe_allow_html=True)

    # Probability bars
    st.markdown("""
    <div class="prob-container">
        <div class="section-title">📊 Prediction Probabilities</div>
    """, unsafe_allow_html=True)

    for cls, prob in probabilities.items():
        pct = prob * 100
        color = "#38ef7d" if cls == "Negative" else "#ff416c"
        st.markdown(f"""
        <div class="prob-label">{cls}</div>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{pct}%; background:{color};">
                {pct:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Show input summary in an expander
    with st.expander("📝 View Input Summary"):
        summary_cols = st.columns(3)
        items = list(input_data.items())
        for i, (key, val) in enumerate(items):
            info = FEATURE_INFO.get(key, {})
            with summary_cols[i % 3]:
                st.metric(label=info.get("label", key), value=val)
