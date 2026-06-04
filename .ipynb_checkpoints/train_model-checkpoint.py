import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset
df = pd.read_excel(r"C:\Users\yawer\Downloads\Student-Performance-Prediction\student_data.xlsx")
print(df.columns)
print(df.head())

# Convert Pass/Fail to numbers
df['Result'] = (df['Final_Grade'] >= 80).astype(int)
print(df['Result'].value_counts())
print(df['Final_Grade'].describe())
# Features and target
X = df[['Study_Hours', 'Attendance', 'Previous_Score']]
y = df['Result']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")