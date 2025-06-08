import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = "eda/mental_health_dataset.csv"
PLOT_DIR = "assets/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


df=pd.read_csv(DATA_PATH)

df.columns=df.columns.str.strip().str.lower().str.replace(" ", "_")

print("Shape: ",df.shape)
print("Columns: ",df.columns.tolist())
print(df.head())

print("\n -----Dataset Info------")
print(df.info())

print("------Statistical Summary-----")
print(df.describe())


print("\n----Missing Values-----")
print(df.isnull().sum())

categorical_cols=['gender','employment_status','work_environment','mental_health_history','seeks_treatment','mental_health_risk']
print("\n--Unique values in categorical columns--")
for col in categorical_cols:
    print(f"{col}:{df[col].unique()}")

#Check age range
print(f"\nAge range: {df['age'].min()} to {df['age'].max()}")
#Remove unrealistic ages
df=df[(df['age']>=10)&(df['age']<=100)]
print(f"After removing outliers , dataset shape:{df.shape}")


#Optionally group similar responses (Your can expand this mapping)
df['gender'] = df['gender'].replace({'male':'Male','m':'Male',
                                     'female':'Female','f':'Female',
                                     'non-binary':'Non-binary','nb':'Non-binary'
                                     })

print(f"\nUnique gender values after cleaning:{df['gender'].unique()}")

plt.show()
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='mental_health_risk', hue='mental_health_risk' , palette='Set2',legend=False)
plt.title("Distribution of Mental Health Risk Levels")
os.makedirs("plots", exist_ok=True)  # Ensure folder exists
plt.savefig("assets/plots/mental_health_risk_distribution.png")
plt.show()

#Correlation Heatmap
numeric_cols=df.select_dtypes(include='number')

plt.figure(figsize=(10,8))
sns.heatmap(numeric_cols.corr(),annot=True,cmap='coolwarm',fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()


os.makedirs("assets/plots",exist_ok=True)
plt.savefig("assets/plots/correlation_heatmap.png")
plt.show()

#Box Plot 
plt.figure(figsize=(6,4))
sns.boxplot(data=df,x='mental_health_risk',y='stress_level',palette='Set3')
plt.title('Stress Levels across Mental Health Risk')
plt.savefig('assets/plots/stress_vs_risk.png')
plt.show()

#CountPlot
plt.figure(figsize=(6,4))
sns.countplot(data=df,x='gender',hue='mental_health_risk',palette='pastel')
plt.title('Mental Health Risk Distribution by Gender')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("assets/plots/gender_vs_risk.png")
plt.show()

#Pairplot
sns.pairplot(df[['stress_level','sleep_hours','depression_score','anxiety_score','mental_health_risk']],hue='mental_health_risk',palette='husl')
plt.savefig("assets/plots/pairplot_risk_features.png")
plt.show()

df.to_csv("data/mental_health_cleaned.csv",index=False)

