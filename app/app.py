import streamlit as st
import pandas as pd
import joblib
import os

# Load model artifacts
model = joblib.load("models/mental_health_risk_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
model_features = joblib.load("models/model_features.pkl")

# Sidebar navigation
st.sidebar.header("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🔴 Mental Health Risk Predictor", "📊 Insights Dashboard"])

# Load cleaned data for dashboard use
@st.cache_data
def load_data():
    return pd.read_csv("data/mental_health_cleaned.csv")

df = load_data()

# Page 1: Mental Health Risk Predictor
if page == "🔴 Mental Health Risk Predictor":
    st.title("🧠 Mental Health Risk Predictor")

    with st.form("prediction_form"):
        st.subheader("Enter Details")

        age = st.slider("Age", 15, 70, 30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Student", "Other"])
        work_environment = st.selectbox("Work Environment", ["Supportive", "Neutral", "Toxic"])
        mental_health_history = st.selectbox("Mental Health History", ["Yes", "No"])
        seeks_treatment = st.selectbox("Seeks Treatment", ["Yes", "No"])
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
        sleep_hours = st.slider("Average Sleep (hours)", 0, 12, 6)
        physical_activity_days = st.slider("Exercise Days per Week", 0, 7, 3)
        depression_score = st.slider("Depression Score (1-10)", 1, 10, 5)
        anxiety_score = st.slider("Anxiety Score (1-10)", 1, 10, 5)
        social_support_score = st.slider("Social Support Score (1-10)", 1, 10, 5)
        productivity_score = st.slider("Productivity Score (1-10)", 1, 10, 5)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_dict = {
            "age": age,
            "gender": gender,
            "employment_status": employment_status,
            "work_environment": work_environment,
            "mental_health_history": mental_health_history,
            "seeks_treatment": seeks_treatment,
            "stress_level": stress_level,
            "sleep_hours": sleep_hours,
            "physical_activity_days": physical_activity_days,
            "depression_score": depression_score,
            "anxiety_score": anxiety_score,
            "social_support_score": social_support_score,
            "productivity_score": productivity_score
        }

        df_input = pd.DataFrame([input_dict])
        df_encoded = pd.get_dummies(df_input)

        for col in model_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[model_features]

        scaled_input = scaler.transform(df_encoded)
        pred = model.predict(scaled_input)
        risk = label_encoder.inverse_transform(pred)[0]

        st.success(f"💡 Predicted Mental Health Risk: **{risk}**")

# Page 2: Insights Dashboard
elif page == "📊 Insights Dashboard":
    st.title("📊 Mental Health Insights Dashboard")

    st.subheader("🔹 Overall Risk Distribution")
    st.image("assets/plots/mental_health_risk_distribution.png", use_column_width=True)

    st.subheader("🔹 Gender vs Risk")
    st.image("assets/plots/gender_vs_risk.png", use_column_width=True)

    st.subheader("🔹 Stress vs Risk")
    st.image("assets/plots/stress_vs_risk.png", use_column_width=True)

    st.subheader("🔹 Correlation Heatmap")
    st.image("assets/plots/correlation_heatmap.png", use_column_width=True)

    st.subheader("🔹 Pairplot of Key Features")
    st.image("assets/plots/pairplot_risk_features.png", use_column_width=True)

    st.markdown("📝 *These insights help identify key patterns and signals for predicting mental health risk.*")
