import pandas as pd
import joblib

#Load model and preprocessing tools
model=joblib.load("models/mental_health_risk_model.pkl")
scaler=joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

sample= {
    'age': 29,
    'stress_level':7,
    'sleep_hours':5,
    'physical_activity_days':1,
    'depression_score':8,
    'anxiety_score':7,
    'social_support_score':3,
    'productivity_score':4,
    'gender':'Male',
    'employement_status':'Employed',
    'work_environmet':'Hostile',
    'mental_health_history':'Yes',
    'seeks_treatment':'No'
}

#Convert To DataFrame
df=pd.DataFrame([sample])

#One-hot encoding for categorical features
df_encoded = pd.get_dummies(df)

#Allign with training features(add missing col if needed)
model_features=joblib.load("models/model_features.pkl")
for col in model_features:
    if col not in df_encoded.columns:
        df_encoded[col]=0

df_encoded=df_encoded[model_features]

#Scale
df_scaled=scaler.transform(df_encoded)

#Predict
pred=model.predict(df_scaled)
pred_label=label_encoder.inverse_transform(pred)

print("Predicted Mental Health Risk:",pred_label[0])

