import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

df=pd.read_csv("data/mental_health_cleaned.csv")

label_encoder=LabelEncoder()
df['mental_health_risk']=label_encoder.fit_transform(df['mental_health_risk'])


os.makedirs("models",exist_ok=True)
joblib.dump(label_encoder,"models/label_encoder.pkl")


#Split features target
X=df.drop("mental_health_risk",axis=1)
y=df["mental_health_risk"]

X=pd.get_dummies(X)
joblib.dump(X.columns.tolist(), "models/model_features.pkl")


#Scale Feature
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

#Save Scaler
joblib.dump(scaler,"models/scaler.pkl")

#Train/ Test Split
X_train,X_test,y_train,y_test= train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Train Model
model= RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Predict
y_pred=model.predict(X_test)

#Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred))

#Save Model
joblib.dump(model,"models/mental_health_risk_model.pkl")



