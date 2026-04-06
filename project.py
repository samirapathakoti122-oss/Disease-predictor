import streamlit as st
import pickle
import os
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# STEP 1: CREATE MODEL IF NOT EXISTS
# -------------------------------
model_path = "model.pkl"

if not os.path.exists(model_path):
    # Sample training data
    X = [
        [1,1,1,1,0,0],
        [1,1,0,1,1,0],
        [0,1,1,0,0,1],
        [0,0,0,1,1,1],
        [1,0,1,1,0,1]
    ]

    y = ["Flu", "Cold", "Migraine", "Food Poisoning", "Viral Fever"]

    # Train model
    model = RandomForestClassifier()
    model.fit(X, y)

    # Save model
    with open(model_path, "wb") as file:
        pickle.dump(model, file)

# -------------------------------
# STEP 2: LOAD MODEL
# -------------------------------
with open(model_path, "rb") as file:
    model = pickle.load(file)

# -------------------------------
# STEP 3: STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Disease Predictor", page_icon="🩺")

st.title("🩺 Disease Prediction System")

st.write("Select symptoms (0 = No, 1 = Yes)")

fever = st.selectbox("Fever", [0, 1])
cough = st.selectbox("Cough", [0, 1])
headache = st.selectbox("Headache", [0, 1])
fatigue = st.selectbox("Fatigue", [0, 1])
vomiting = st.selectbox("Vomiting", [0, 1])
cold = st.selectbox("Cold", [0, 1])

# -------------------------------
# STEP 4: PREDICTION
# -------------------------------
if st.button("Predict"):
    try:
        input_data = [[fever, cough, headache, fatigue, vomiting, cold]]
        result = model.predict(input_data)
        st.success(f"✅ Predicted Disease: {result[0]}")
    except Exception as e:
        st.error(f"❌ Error: {e}")