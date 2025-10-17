import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Zepto Review Classifier", layout="wide")

# Load model and data
model = joblib.load("models/best_model.pkl")
data = pd.read_csv("data/final_dataset-2.csv")

st.title("🛒 Zepto Fake Review Classifier")

# Input
review = st.text_input("Enter review text:")
if review:
    # Dummy prediction logic (replace with actual)
    st.write("Prediction:", "Fake" if "bad" in review.lower() else "Genuine")

    # SHAP explanation
    explainer = shap.Explainer(model, data.drop(columns=["target"]))
    shap_values = explainer(data.iloc[[0]])
    st.subheader("SHAP Explanation")
    fig = shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
