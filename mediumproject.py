import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# 1. DATASET
# -------------------------
data = {
    "StudyHours": [1,2,3,4,5,6,7,2,3,4,6,7,8,1,5],
    "Attendance": [50,60,65,70,75,80,90,55,68,72,85,88,95,45,78],
    "Sleep": [5,6,6,7,7,8,8,5,6,7,7,8,8,4,7],
    "SocialMedia": [5,4,3,3,2,2,1,5,4,3,2,1,1,6,2],
    "Result": ["Fail","Fail","Average","Average","Good","Good","Good",
               "Fail","Average","Average","Good","Good","Good","Fail","Good"]
}

df = pd.DataFrame(data)

# -------------------------
# 2. PREPROCESS
# -------------------------
X = df[["StudyHours","Attendance","Sleep","SocialMedia"]]
y = df["Result"].map({"Fail":0,"Average":1,"Good":2})

# -------------------------
# 3. MODEL
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# -------------------------
# 4. STREAMLIT UI
# -------------------------
st.title("📚 Student Performance Predictor AI")

study = st.slider("Study Hours", 0, 10)
attendance = st.slider("Attendance %", 0, 100)
sleep = st.slider("Sleep Hours", 3, 10)
social = st.slider("Social Media Usage (hrs)", 0, 8)

if st.button("Predict Performance"):
    result = model.predict([[study, attendance, sleep, social]])

    if result[0] == 0:
        st.error("❌ Poor Performance")
    elif result[0] == 1:
        st.warning("⚠️ Average Performance")
    else:
        st.success("✅ Good Performance")

    # -------------------------
    # SMART SUGGESTIONS
    # -------------------------
    st.subheader("📌 Suggestions")

    if study < 4:
        st.info("Increase study hours 📖")

    if attendance < 75:
        st.info("Improve attendance 🏫")

    if sleep < 6:
        st.info("Get enough sleep 😴")

    if social > 4:
        st.info("Reduce social media usage 📱")
