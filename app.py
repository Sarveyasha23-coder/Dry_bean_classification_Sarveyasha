import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Dry Bean Classification System",
    page_icon="🌱",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_files():
    model = joblib.load("happyglad.pkl")
    scaler = joblib.load("bean_scaler.pkl")
    return model, scaler

model, scaler = load_files()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 3.2em;
        font-size: 18px;
        font-weight: 600;
        background-color: #2E8B57;
        color: white;
    }
    .stButton > button:hover {
        background-color: #246B45;
        color: white;
    }
    .creator-box {
        background-color: #f7f9fc;
        padding: 12px;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        font-size: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("🌱 Smart Dry Bean Classification System")
st.markdown("### AI-Powered Bean Variety Prediction using Machine Learning")

st.markdown("""
<div class="creator-box">
Created by <b>Sarveyasha Sodhiya</b> • B.Tech (AI & ML) Portfolio Project
</div>
""", unsafe_allow_html=True)

st.write("Enter the bean feature values below to predict the bean class accurately.")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📌 Project Overview")
    st.info(
        "This machine learning application predicts the class of dry beans "
        "based on 16 morphological features using a trained classification model."
    )

# ---------------- INPUT FIELDS ----------------
feature_names = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRatio", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "Roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"
]

st.subheader("🔢 Enter Bean Measurements")

col1, col2 = st.columns(2)
features = []

for i, feature in enumerate(feature_names):
    if i % 2 == 0:
        with col1:
            value = st.number_input(feature, min_value=0.0, value=0.0, format="%.4f")
    else:
        with col2:
            value = st.number_input(feature, min_value=0.0, value=0.0, format="%.4f")
    features.append(value)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Bean Type"):
    try:
        input_data = np.array(features).reshape(1, -1)
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)

        bean_labels = ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA']

        raw_class = int(prediction[0])
        predicted_label = bean_labels[raw_class]

        st.success(f"✅ Predicted Bean Type: {predicted_label}")

        # Display entered values
        df = pd.DataFrame({
            "Feature": feature_names,
            "Entered Value": features
        })

        with st.expander("📊 View Input Summary"):
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with Streamlit, Scikit-learn and Python | Machine Learning Portfolio Project by Sarveyasha Sodhiya")
