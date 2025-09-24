import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- Load the trained model ---
model = joblib.load("heart_disease_model.pkl")

# --- Layout with image beside title ---
col1, col2 = st.columns([1, 4])
with col1:
    st.image("baymaxtitle.png", width=100)  
with col2:
    st.title("BAYMAX_0.1")
    st.subheader("BAYMAX here to help you with your heart")

st.write("Enter patient details to check the risk of heart disease.")

# --- Input fields with styled ranges ---
age = st.number_input("Age", min_value=0, max_value=200, value=50)


sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")


cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
st.markdown("<span style='color:gray; font-size:12px; font-family:Courier'>Types: 1–4</span>", unsafe_allow_html=True)

trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
st.markdown("<span style='color:gray; font-size:12px; font-family:Courier'>Range: 80–200 mmHg</span>", unsafe_allow_html=True)

chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
st.markdown("<span style='color:gray; font-size:12px; font-family:Courier'>Range: 100–600 mg/dl</span>", unsafe_allow_html=True)

blood_sugar = st.number_input("Fasting Blood Sugar (mg/dl)", min_value=50, max_value=400, value=100)
st.markdown("<span style='color:gray; font-size:12px; font-family:Courier'>Range: 50–400 mg/dl </span>", unsafe_allow_html=True)
fbs = 1 if blood_sugar > 120 else 0

restecg = st.selectbox("Resting Electrocardiographic", [0, 1, 2])
st.markdown("<span style='color:gray; font-size:12px; font-family:Courier'>Values: 0, 1, 2</span>", unsafe_allow_html=True)

thalach = st.number_input("Maximum Heart Rate Achieved", min_value=70, max_value=220, value=150)
st.markdown("<span style='color:gray; font-size:12px; font-family:Courier'>Range: 70–220 bpm</span>", unsafe_allow_html=True)

slope = st.selectbox("Slope of the Peak Exercise ST Segment", [1, 2, 3])
st.markdown("<span style='color:gray; font-size:12px; font-family:Courier'>Values: 1–3</span>", unsafe_allow_html=True)

exang_input = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
exang = 1 if exang_input == "Yes" else 0

thal = st.selectbox("Thalassemia", [3, 6, 7])
st.markdown("<span style='color:gray; font-size:12px; font-family:Courier'>Values: 3, 6, 7</span>", unsafe_allow_html=True)

oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
st.markdown("<span style='color:gray; font-size:12px; font-family:Courier'>Range: 0.0–7.0</span>", unsafe_allow_html=True)

ca = st.number_input("Number of Major Vessels", min_value=0, max_value=3, value=0)
st.markdown("<span style='color:gray; font-size:12px; font-family:Courier'>Range: 0–3 vessels</span>", unsafe_allow_html=True)

# --- Encode categorical features as one-hot ---
features = np.zeros(22)
features[0] = age
features[1] = sex
features[2] = 1 if cp == 1 else 0
features[3] = 1 if cp == 2 else 0
features[4] = 1 if cp == 3 else 0
features[5] = 1 if cp == 4 else 0
features[6] = trestbps
features[7] = chol
features[8] = fbs
features[9]  = 1 if restecg == 0 else 0
features[10] = 1 if restecg == 1 else 0
features[11] = 1 if restecg == 2 else 0
features[12] = thalach
features[13] = 1 if slope == 1 else 0
features[14] = 1 if slope == 2 else 0
features[15] = 1 if slope == 3 else 0
features[16] = exang
features[17] = 1 if thal == 3 else 0
features[18] = 1 if thal == 6 else 0
features[19] = 1 if thal == 7 else 0
features[20] = oldpeak
features[21] = ca

# --- Prediction ---
if st.button("Predict"):
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0][1]

    st.session_state["prediction"] = prediction
    st.session_state["probability"] = probability

    st.write(f"Predicted probability of heart disease: {probability*100:.2f}%")

    if prediction == 1:
        st.error("⚠️ High risk of heart disease detected.")
        st.markdown("<h4 style='color:red'>Please consult a doctor immediately. 🩺</h4>", unsafe_allow_html=True)
        st.image("lollipop.png", width=300)
    else:
        st.success("✅ Low risk of heart disease.")
        st.markdown("<h4 style='color:green'>Keep maintaining a healthy lifestyle! 💪</h4>", unsafe_allow_html=True)
        st.image("lollipop.png", width=300)

# --- Show Statistics (only appears after prediction) ---
if "prediction" in st.session_state:
    if st.button("Show Statistics 📊"):
        st.subheader("📈 Statistics Section")
        st.write("Here you can explore charts, model performance metrics, or patient history.")

        df = pd.read_csv("data/heart_disease_clean.csv") 
        label_col = df.columns[-1]

        # Example 1: Count plot
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=label_col, palette="Set2", ax=ax)
        ax.set_title("Heart Disease Distribution in Dataset")
        st.pyplot(fig)

        # Example 2: Age distribution
        if "age" in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(data=df, x="age", hue=label_col, multiple="stack", bins=20, palette="Set2", ax=ax)
            ax.set_title("Age Distribution by Heart Disease Status")
            st.pyplot(fig)

        # Example 3: Correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = df.corr()
        sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap of Features")
        st.pyplot(fig)

        # Example 4: Pie chart of this specific prediction
        labels = ["High Risk", "Low Risk"]
        values = [st.session_state["prediction"], 1 - st.session_state["prediction"]]

        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct='%1.1f%%')
        ax.set_title("Prediction Outcome Distribution")
        st.pyplot(fig)
