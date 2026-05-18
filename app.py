import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Dry Bean Classification",
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

# ---------------- CUSTOM STYLE ----------------
st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🌱 Dry Bean Classification System")
st.write("Predict the type of dry bean using machine learning")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Project Info")
st.sidebar.info(
    "This app predicts dry bean variety based on 16 physical measurements."
)

# ---------------- FEATURE INPUT ----------------
feature_names = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRatio", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "Roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"
]

st.subheader("Enter Bean Features")

col1, col2 = st.columns(2)
features = []

for i, feature in enumerate(feature_names):
    if i % 2 == 0:
        with col1:
            val = st.number_input(feature, min_value=0.0, format="%.4f")
    else:
        with col2:
            val = st.number_input(feature, min_value=0.0, format="%.4f")
    features.append(val)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Bean Type"):
    try:
        input_data = np.array(features).reshape(1, -1)
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)

        bean_labels = ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA']

        predicted_label = bean_labels[int(prediction[0])]

        st.success(f"✅ Predicted Bean Type: {predicted_label}")

        # Show entered values
        df = pd.DataFrame({
            "Feature": feature_names,
            "Input Value": features
        })

        st.subheader("Input Summary")
        st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with Streamlit and Machine Learning")
