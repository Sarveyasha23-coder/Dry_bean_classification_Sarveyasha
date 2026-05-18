import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Page config
st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
.main {padding-top: 1rem;}
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}
.metric-card {
    background-color: #f6f8fa;
    padding: 1rem;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
# Load model once

def load_assets():
    model = joblib.load("happyglad.pkl")
    scaler = joblib.load("bean_scaler.pkl")
    return model, scaler

model, scaler = load_assets()

st.title("🌱 Smart Dry Bean Classification System")
st.caption("AI-powered bean variety prediction using machine learning")

with st.sidebar:
    st.header("About")
    st.write("Enter the 16 numerical bean features to classify the bean type.")
    st.info("Tip: Use realistic dataset values for better predictions.")

feature_names = [
    "Area", "Perimeter", "Major Axis Length", "Minor Axis Length",
    "Aspect Ratio", "Eccentricity", "Convex Area", "Equivalent Diameter",
    "Extent", "Solidity", "Roundness", "Compactness",
    "Shape Factor 1", "Shape Factor 2", "Shape Factor 3", "Shape Factor 4"
]

st.subheader("Enter Bean Measurements")
cols = st.columns(2)
features = []

for i, name in enumerate(feature_names):
    with cols[i % 2]:
        val = st.number_input(name, min_value=0.0, value=0.0, format="%.4f")
        features.append(val)

col1, col2 = st.columns([3, 1])
with col1:
    predict = st.button("🔍 Predict Bean Type")
with col2:
    clear = st.button("🔄 Reset")

if predict:
    try:
        features_array = np.array(features).reshape(1, -1)
        scaled = scaler.transform(features_array)
        prediction = model.predict(scaled)[0]

        st.success(f"✅ Predicted Bean Class: **{prediction}**")

        df = pd.DataFrame({"Feature": feature_names, "Value": features})
        with st.expander("View Input Summary"):
            st.dataframe(df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Built with Streamlit • Machine Learning Project Portfolio")
