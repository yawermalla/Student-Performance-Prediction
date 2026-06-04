import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset
df = pd.read_excel("student_data.xlsx")

# Explore dataset
print("Columns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

# Create target variable
# Pass = 1 if Final_Grade >= 80, otherwise Fail = 0
df["Result"] = (df["Final_Grade"] >= 80).astype(int)

print("\nResult Distribution:")
print(df["Result"].value_counts())

# Features and Target
X = df[["Study_Hours", "Attendance", "Previous_Score"]]
y = df["Result"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = LogisticRegression()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Performance")
print("-" * 30)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save Model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved successfully as 'model.pkl'")