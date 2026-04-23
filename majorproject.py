# 🚀 FULL AI/ML PROJECT IN ONE CELL

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# -----------------------------
# 1. CREATE SMALL DATASET (15 rows)
# -----------------------------
data = {
    "Age": [18, 22, 25, 30, 35, 40, 28, 32, 45, 50, 23, 27, 38, 42, 29],
    "BMI": [19, 22, 27, 30, 31, 29, 24, 26, 32, 34, 21, 23, 28, 33, 25],
    "Steps": [8000, 6000, 4000, 3000, 2000, 3500, 7000, 6500, 2500, 1500, 9000, 7500, 5000, 2000, 6800],
    "Sleep": [7, 6, 5, 5, 4, 6, 7, 6, 4, 3, 8, 7, 5, 4, 6],
    "Water": [3, 2.5, 2, 1.5, 1.2, 2, 3, 2.8, 1.5, 1, 3.5, 3, 2, 1.3, 2.7],
    "Risk": ["Low", "Low", "Medium", "High", "High", "Medium", "Low", "Medium", "High", "High", "Low", "Low", "Medium", "High", "Medium"]
}

df = pd.DataFrame(data)

# -----------------------------
# 2. PREPROCESS
# -----------------------------
X = df[["Age", "BMI", "Steps", "Sleep", "Water"]]
y = df["Risk"]

# Convert labels to numbers
y = y.map({"Low": 0, "Medium": 1, "High": 2})

# -----------------------------
# 3. TRAIN MODEL
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# -----------------------------
# 4. ACCURACY
# -----------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# -----------------------------
# 5. STREAMLIT APP
# -----------------------------
st.title("🏋️‍♀️ Fitness Risk Predictor AI")

st.write(f"Model Accuracy: {acc:.2f}")

age = st.slider("Age", 10, 60)
bmi = st.slider("BMI", 15, 40)
steps = st.slider("Daily Steps", 1000, 10000)
sleep = st.slider("Sleep Hours", 3, 10)
water = st.slider("Water Intake (litres)", 1.0, 4.0)

if st.button("Predict Risk"):
    result = model.predict([[age, bmi, steps, sleep, water]])

    if result[0] == 0:
        st.success("🟢 Low Risk")
    elif result[0] == 1:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

    # Suggestions (🔥 important for marks)
    if steps < 5000:
        st.info("Increase daily steps 🚶")
    if sleep < 6:
        st.info("Sleep more 😴")
    if water < 2:
        st.info("Drink more water 💧")
