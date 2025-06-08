# 🧠 Mental Health Risk Predictor

A **Streamlit-based machine learning app** that predicts mental health risk and visualizes insights using a real-world dataset. This project explores mental health trends, identifies risk factors, and provides an interactive interface for predictions.

## 🚀 Live Demo
👉 [Click here to try the app](https://share.streamlit.io/your-a700sharma/mental-health-risk-predictor/main/app/app.py)

---

## 📌 Table of Contents
- [About the Project](#about-the-project
                        A Streamlit-based app that predicts mental health risk and visualizes 
                        insights using a real-world dataset.
                        )
- [Technologies Used](#Technologies Used
                            Python 3.8+
                            Pandas, NumPy
                            Matplotlib, Seaborn
                            Scikit-learn
                            Streamlit
                            Pickle)
- [How to Run Locally](#how-to-run-locally
                       git clone https://github.com/a700sharma/mental-health-risk-predictor.git
                       cd mental-health-risk-predictor
                        )
- [Screenshots](#screenshots)
- [License](#license)
- [Contact](Author: a700sharma

            LinkedIn: [www.linkedin.com/in/asharma700]

            Portfolio: [Your Portfolio Website])

---

## 📖 About the Project

Mental health in the workplace is a growing concern. This project:
- Analyzes workplace mental health data.
- Performs Exploratory Data Analysis (EDA) to uncover trends.
- Builds a machine learning model to predict mental health risk.
- Provides an interactive web app using **Streamlit**.

---

## 🗂 Project Structure

```bash
mental-health-risk-predictor/
│
├── app/                         # Streamlit UI logic
├── assets/plots/               # Plot images used in app
├── data/
│   ├── mental_health_cleaned.csv
│   └── mental_health_dataset.csv
├── eda/
│   └── mental_health_eda.py     # Exploratory data analysis
├── models/
│   ├── model_pipeline.py
│   ├── model_features.pkl
│   ├── mental_health_risk_model.pkl
│   └── scaler.pkl
├── plots/                      # Generated visualization outputs
├── predict.py                  # Prediction backend logic
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
