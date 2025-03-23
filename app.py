import streamlit as st
import joblib  # Use joblib to load .sav models
import numpy as np
from streamlit_option_menu import option_menu

# Set page title and icon
st.set_page_config(page_title="Disease Prediction", page_icon="🧑‍⚕️")

# Hide Streamlit UI elements
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Custom CSS for Inter Font
inter_font_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}
</style>
"""
st.markdown(inter_font_css, unsafe_allow_html=True)

# Background image
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://iaar.chiba-u.jp/en/static/images/belong_detail03.jpg");
    background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load Both Models
try:
    diabetes_model = joblib.load("diabetes.sav")  # Load Diabetes Model
    
except FileNotFoundError:
    st.error("⚠️ Model files missing! Make sure 'diabetes.sav' and 'lung_cancer.sav' exist.")

# Sidebar Menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Predict Diabetes", "About"],
        icons=["house", "syringe", "info-circle"],
        menu_icon="cast",
        default_index=1,
    )

# Home Page
if selected == "Home":
    st.title("🏥 Welcome to Disease Prediction App")
    st.write("This app helps predict **Diabetes** and **Lung Cancer** using trained AI models. Select a disease from the sidebar.")

# Diabetes Prediction Page
elif selected == "Predict Diabetes":
    st.title("🩺 Diabetes Risk Prediction")

    # User input fields
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    HbA1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.7)
    blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)

    # Prepare input data
    input_data = np.array([[age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level]])

    # Predict using the correct model
    if st.button("Predict"):
        if diabetes_model:
            prediction = diabetes_model.predict(input_data)[0]
            if prediction == 1:
                st.error("⚠️ High Risk of Diabetes! Consult a doctor.")
            else:
                st.success("✅ Low Risk of Diabetes! Maintain a healthy lifestyle.")
        else:
            st.error("⚠️ Diabetes model is missing! Cannot make predictions.")



# About Page
elif selected == "About":
    st.title("ℹ️ About This App")
    st.write("""
    This app uses Machine Learning models trained on medical data to predict the risk of **Diabetes** and **Lung Cancer**.
    
    **How it Works:**
    - Select the disease you want to check.
    - Enter your health details in the input fields.
    - Click "Predict" to get an AI-generated risk assessment.
    
    **Disclaimer:** This is an AI-based prediction and should not replace medical advice. Consult a healthcare professional for accurate diagnosis.
    """)
