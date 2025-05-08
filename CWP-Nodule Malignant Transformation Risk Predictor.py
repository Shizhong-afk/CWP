import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mortality Risk Predictor", layout="centered")

st.markdown("""
    <style>
        html, body { font-family: 'Segoe UI', sans-serif; }
        h1 { font-size: 28px !important; text-align: center; color: #2c3e50; }
        h2, h3 { font-size: 20px !important; color: #34495e; }
        .stButton button {
            background-color: #3498db; color: white; border-radius: 8px; padding: 0.5em 1em;
        }
        .stButton button:hover { background-color: #2980b9; }
    </style>
""", unsafe_allow_html=True)

st.title("CWP-Nodule Malignant Transformation Risk Predictor")

model = joblib.load('rf.pkl')

feature_ranges = {
    "MinCT": {"type": "numerical", "min": -1000.0, "max": 100.0, "default": -912.0, "unit": "HU"},
    "Weight": {"type": "numerical", "min": 0.0, "max": 3000.0, "default": 653.3, "unit": "mg"},
    "V": {"type": "numerical", "min": 0.1, "max": 5000.0, "default": 1.87, "unit": "mm³"},
    "AD": {"type": "numerical", "min": 0.1, "max": 60.0, "default": 17.3, "unit": "mm"},
    "Age": {"type": "numerical", "min": 20, "max": 100, "default": 74, "unit": "years"},
    "MaxCA": {"type": "numerical", "min": 0.1, "max": 1000.0, "default": 80.0, "unit": "mm²"}
}

st.markdown("### 🧪 Please input the patient's clinical features:")
feature_values = []
cols = st.columns(2)
for idx, (feature, props) in enumerate(feature_ranges.items()):
    unit = f" ({props['unit']})" if props.get("unit") else ""
    label = f"{feature}{unit} [{props['min']} - {props['max']}]"
    with cols[idx % 2]:
        value = st.number_input(
            label=label,
            min_value=float(props["min"]),
            max_value=float(props["max"]),
            value=float(props["default"]),
            key=feature
        )
        feature_values.append(value)

features = np.array([feature_values])
df_input = pd.DataFrame(features, columns=feature_ranges.keys())

if st.button("🔍 Predict Risk"):
    predicted_class = model.predict(df_input)[0]
    predicted_proba = model.predict_proba(df_input)[0]
    class_names = {0: "🟢 Low Risk (Benign)", 1: "🔴 High Risk (Malignant)"}
    class_name = class_names[predicted_class]
    probability = predicted_proba[predicted_class] * 100

    st.markdown("### 📊 Prediction Result")
    st.success(f"**{class_name}**\n\nProbability: **{int(round(probability))}%**")

    st.markdown("### 🧠 SHAP Feature Contribution")

    # 解释器和 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_input)[predicted_class]

    # 生成 SHAP 力图
    shap.initjs()
    force_plot = shap.force_plot(
        explainer.expected_value[predicted_class],
        shap_values,
        df_input,
        matplotlib=True
    )

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200_