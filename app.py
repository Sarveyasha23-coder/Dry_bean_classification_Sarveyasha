import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

# ---------- LOAD FILES ----------
@st.cache_resource
def load_files():
    model = joblib.load("happyglad.pkl")
    scaler = joblib.load("bean_scaler.pkl")
    return model, scaler

model, scaler = load_files()

# ---------- STYLE ----------
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}
.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.title("🌱 Dry Bean Classification System")
st.write("Predict dry bean type using machine learning")

# ---------- SIDEBAR ----------
st.sidebar.header("About Project")
st.sidebar.info(
    "Enter 16 feature values from the dry bean dataset to predict bean variety."
)

# ---------- INPUT ----------
feature_names = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRatio", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "Roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"
]

st.subheader("Enter Feature Values")

col1, col2 = st.columns(2)
features = []

for i, feature in enumerate(feature_names):
    default_value = 0.0
    if i % 2 == 0:
        with col1:
            val = st.number_input(feature, min_value=0.0, value=default_value, format="%.4f")
    else:
        with col2:
            val = st.number_input(feature, min_value=0.0, value=default_value, format="%.4f")
    features.append(val)

# ---------- PREDICTION ----------
if st.button("🔍 Predict"):
    try:
        input_data = np.array(features).reshape(1, -1)
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)

        # change this list only if your training label order was different
        bean_labels = ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA']

        raw_class = int(prediction[0])

        st.write("Raw predicted class number:", raw_class)

        if 0 <= raw_class < len(bean_labels):
            predicted_label = bean_labels[raw_class]
            st.success(f"✅ Predicted Bean Type: {predicted_label}")
        else:
            st.warning("Prediction class is outside expected range. Check label encoding.")

        # Input summary
        df = pd.DataFrame({
            "Feature": feature_names,
            "Entered Value": features
        })

        with st.expander("View Input Summary"):
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Built with Streamlit • Dry Bean ML Project")
