import streamlit as st
import pickle
import numpy as np
import time
import base64
import matplotlib.pyplot as plt

# Load model
def load_model(path):
    with open(path, 'rb') as file:
        return pickle.load(file)
    
    # Brain stroke
def generate_stroke_report(age, hypertension, avg_glucose_level, bmi, prediction):
    result = "HIGH RISK" if prediction >= 0.7 else "LOW RISK"
    report = f"""
🧾 Brain Stroke Risk Assessment Report

Age: {age}
Hypertension: {'Yes' if hypertension else 'No'}
Average Glucose Level: {avg_glucose_level} mg/dL
BMI: {bmi}

Prediction Result: {result}

Recommendation:
{"Consult a doctor immediately and start lifestyle changes." if prediction >= 0.7 else "Maintain a healthy lifestyle and monitor your health regularly."}
"""
    return report

# Report download
def download_button(text, filename="brain_stroke_report.txt"):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">📄 Download Report</a>'
    return href

# Chart for Stroke Risk Progression
def plot_stroke_risk_progression(age, hypertension, avg_glucose_level):
    age_range = list(range(18, 101, 5))
    risk_scores = []

    for a in age_range:
        score = 0
        score += 1 if a > 60 else 0
        score += hypertension * 1
        score += (avg_glucose_level > 150) * 1
        risk_scores.append(score / 3)  # normalized risk

    fig, ax = plt.subplots()
    ax.plot(age_range, [round(r * 100, 1) for r in risk_scores], marker="o", color="blue")
    ax.set_title("Estimated Stroke Risk Progression Over Age")
    ax.set_xlabel("Age")
    ax.set_ylabel("Estimated Risk (%)")
    st.pyplot(fig)

# Modified Stroke Form with Report and Chart
def stroke_form():
    st.subheader("🧠 Brain Stroke Prediction")

    age = st.slider("Select Age", min_value=18, max_value=100, value=30, step=1)
    
    hypertension = st.selectbox("Do you have Hypertension?", ["No", "Yes"])
    hypertension = 1 if hypertension == "Yes" else 0

    avg_glucose_level = st.slider("Average Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, value=100.0, step=0.5)

    bmi = st.slider("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=22.0, step=0.1)

    if st.button("🔍 Predict Stroke Risk"):
        model = load_model("models/brain-stroke_model.pkl")
        input_data = np.array([[age, hypertension, avg_glucose_level, bmi]])

        try:
            probability = model.predict_proba(input_data)[0][1]  # Probability of class '1' (stroke)
            percentage = round(probability * 100, 2)

            st.info(f"🧪 There is a **{percentage}% chance** of having a stroke.")

            if percentage >= 70:
                st.warning("⚠️ High Risk! Consult a doctor immediately. Start lifestyle changes today.")
                st.write("🕒 You should aim to improve within the next **3-6 months** by maintaining a healthy diet, reducing glucose levels, and managing blood pressure.")
            elif 40 <= percentage < 70:
                st.info("⚠️ Moderate Risk. It's a good time to make positive changes.")
                st.write("🕒 Try to improve your health in the next **6-12 months** through regular checkups, improved diet, and exercise.")
            else:
                st.success("✅ Low Risk! Keep maintaining a healthy lifestyle.")
                st.write("💡 Regular monitoring is still recommended every 6 months.")

            # Stroke Risk chart
            plot_stroke_risk_progression(age, hypertension, avg_glucose_level)

            # Report
            report = generate_stroke_report(age, hypertension, avg_glucose_level, bmi, probability)
            st.markdown(download_button(report), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error in prediction: {e}")

# Breast Cancer 
def generate_breast_cancer_report(age, radius_mean, texture_mean, perimeter_mean, prediction):
    result = "MALIGNANT" if prediction == 1 else "BENIGN"
    report = f"""
🧾 Breast Cancer Detection Report

Age: {age}
Radius Mean: {radius_mean}
Texture Mean: {texture_mean}
Perimeter Mean: {perimeter_mean}

Prediction Result: {result}

Recommendation:
{"Consult a doctor for further tests and treatment." if prediction == 1 else "Continue regular health checkups."}
"""
    return report

# Download Report Button
def download_button(text, filename="breast_cancer_report.txt"):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">📄 Download Report</a>'
    return href

# Risk Chart (based on radius_mean)
def plot_breast_cancer_risk_progression(radius_mean, texture_mean, perimeter_mean):
    feature_range = np.linspace(0, 30, 100)

    risk_scores = []
    for feature in feature_range:
        score = (feature / 30) * 1  # Simulated risk
        risk_scores.append(score)

    fig, ax = plt.subplots()
    ax.plot(feature_range, [round(r * 100, 1) for r in risk_scores], marker="o", color="purple")
    ax.set_title("Estimated Risk Progression Based on Radius Mean")
    ax.set_xlabel("Radius Mean")
    ax.set_ylabel("Estimated Risk (%)")
    st.pyplot(fig)

# Modified Breast Cancer Form with Age
def breast_cancer_form():
    st.subheader("🎗️ Breast Cancer Detection")

    age = st.slider("Age", 20, 100, 40)
    radius_mean = st.slider("Radius Mean", 0.0, 30.0)
    texture_mean = st.slider("Texture Mean", 0.0, 40.0)
    perimeter_mean = st.slider("Perimeter Mean", 0.0, 200.0)

    if st.button("🔍 Predict Breast Cancer"):
        model = pickle.load(open("models/breast-model", "rb"))  # Ensure the model supports 4 features!
        input_data = np.array([[age, radius_mean, texture_mean, perimeter_mean]])
        prediction = model.predict(input_data)[0]

        st.success("Malignant" if prediction == 1 else "Benign")

        plot_breast_cancer_risk_progression(radius_mean, texture_mean, perimeter_mean)

        report = generate_breast_cancer_report(age, radius_mean, texture_mean, perimeter_mean, prediction)
        st.markdown(download_button(report), unsafe_allow_html=True)

# Lung Cancer Report Generator
def generate_lung_cancer_report(age, smoking, coughing, chest_pain, prediction):
    result = "HIGH RISK DETECTED" if prediction == 1 else "LOW RISK"
    report = f"""
🧾 Lung Cancer Screening Report

Symptoms:
- Age: {age}
- Smoking: {'Yes' if smoking else 'No'}
- Coughing: {'Yes' if coughing else 'No'}
- Chest Pain: {'Yes' if chest_pain else 'No'}

Prediction Result: {result}

Recommendation:
{"Please consult a doctor immediately for further tests." if prediction == 1 else "Maintain a healthy lifestyle and consider regular checkups."}
"""
    return report

# Download Button
def download_button(text, filename="lung_cancer_report.txt"):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">📄 Download Report</a>'
    return href

# Risk Chart for Lung Cancer
def plot_lung_cancer_risk(smoking, coughing, chest_pain):
    symptom_counts = list(range(0, 4))
    actual_symptoms = smoking + coughing + chest_pain
    risk_scores = [count / 3 for count in symptom_counts]

    fig, ax = plt.subplots()
    ax.plot(symptom_counts, [r * 100 for r in risk_scores], marker='o', color='red')
    ax.axvline(actual_symptoms, color='gray', linestyle='--', label=f'Your Symptoms: {actual_symptoms}')
    ax.set_title("Lung Cancer Risk Based on Symptoms")
    ax.set_xlabel("Number of Symptoms")
    ax.set_ylabel("Estimated Risk (%)")
    ax.legend()
    st.pyplot(fig)

# Lung Cancer Form
def lung_cancer_form():
    st.subheader("🫁 Lung Cancer Detection")

    age = st.number_input("Age", min_value=10, max_value=100, step=1)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    coughing = st.selectbox("Coughing", ["No", "Yes"])
    chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])

    # Convert to binary
    smoking_val = 1 if smoking == "Yes" else 0
    coughing_val = 1 if coughing == "Yes" else 0
    chest_pain_val = 1 if chest_pain == "Yes" else 0

    if st.button("🔍 Predict Lung Cancer"):
        with st.spinner("Analyzing symptoms..."):
            model = pickle.load(open("models/Lung_model.pkl", "rb"))
            input_data = np.array([[age, smoking_val, coughing_val, chest_pain_val]])
            prediction = model.predict(input_data)[0]

            if prediction == 1:
                st.error("⚠️ High risk of lung cancer symptoms detected.")
            else:
                st.success("✅ Low risk detected.")

            plot_lung_cancer_risk(smoking_val, coughing_val, chest_pain_val)

            report = generate_lung_cancer_report(age, smoking_val, coughing_val, chest_pain_val, prediction)
            st.markdown(download_button(report), unsafe_allow_html=True)

                
# Early Cancer Input
# Report generator for early cancer
def generate_early_cancer_report(weight_loss, pain, fatigue, fever, prediction):
    result = "POSSIBLE SYMPTOMS DETECTED" if prediction == 1 else "NO STRONG SIGNS DETECTED"
    report = f"""
🧾 Early Cancer Screening Report

Symptoms:
- Weight Loss: {'Yes' if weight_loss else 'No'}
- Pain: {'Yes' if pain else 'No'}
- Fatigue: {'Yes' if fatigue else 'No'}
- Fever: {'Yes' if fever else 'No'}

Prediction Result: {result}

Recommendation:
{"Consult a doctor for further examination." if prediction == 1 else "Maintain a healthy lifestyle and monitor symptoms regularly."}
"""
    return report

# Reuse download button
def download_button(text, filename="early_cancer_report.txt"):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">📄 Download Report</a>'
    return href

# Risk chart based on number of symptoms
def plot_early_cancer_risk(weight_loss, pain, fatigue, fever):
    symptom_counts = list(range(0, 5))
    actual_symptoms = weight_loss + pain + fatigue + fever

    risk_scores = [count / 4 for count in symptom_counts]

    fig, ax = plt.subplots()
    ax.plot(symptom_counts, [r * 100 for r in risk_scores], marker='o', color='orange')
    ax.axvline(actual_symptoms, color='gray', linestyle='--', label=f'Your Symptoms: {actual_symptoms}')
    ax.set_title("Risk Estimation Based on Number of Symptoms")
    ax.set_xlabel("Number of Symptoms")
    ax.set_ylabel("Estimated Risk (%)")
    ax.legend()
    st.pyplot(fig)

# Main Form
def early_cancer_form():
    st.subheader("🧫 Early Cancer Detection")

    weight_loss = st.selectbox("Weight Loss", ["No", "Yes"])
    pain = st.selectbox("Pain", ["No", "Yes"])
    fatigue = st.selectbox("Fatigue", ["No", "Yes"])
    fever = st.selectbox("Fever", ["No", "Yes"])

    # Convert to binary
    weight_loss_val = 1 if weight_loss == "Yes" else 0
    pain_val = 1 if pain == "Yes" else 0
    fatigue_val = 1 if fatigue == "Yes" else 0
    fever_val = 1 if fever == "Yes" else 0

    if st.button("🔍 Predict Early Cancer"):
        with st.spinner("Analyzing symptoms..."):
            model = load_model("models/Early-cancer.pkl")
            input_data = np.array([[weight_loss_val, pain_val, fatigue_val, fever_val]])
            prediction = model.predict(input_data)[0]

            if prediction == 1:
                st.error("⚠️ Possible early cancer symptoms detected.")
            else:
                st.success("✅ No strong signs detected.")

            # Risk chart
            plot_early_cancer_risk(weight_loss_val, pain_val, fatigue_val, fever_val)

            # Report
            report = generate_early_cancer_report(weight_loss_val, pain_val, fatigue_val, fever_val, prediction)
            st.markdown(download_button(report), unsafe_allow_html=True)
# Main App
st.title("🧬 Multi-Disease Prediction System")

choice = st.sidebar.selectbox("Choose a Prediction Model", [
    "Brain Stroke Prediction",
    "Breast Cancer Detection",
    "Lung Cancer Detection",
    "Early Cancer Prediction"
])

if choice == "Brain Stroke Prediction":
    stroke_form()
elif choice == "Breast Cancer Detection":
    breast_cancer_form()
elif choice == "Lung Cancer Detection":
    lung_cancer_form()
elif choice == "Early Cancer Prediction":
    early_cancer_form()
