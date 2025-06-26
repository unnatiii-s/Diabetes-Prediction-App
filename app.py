<<<<<<< HEAD
#importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
from tensorflow import keras
import joblib
import matplotlib.pyplot as plt
import streamlit as st

model = tf.keras.models.load_model('Diabetic_model.h5')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("Diabetes Prediction App")
st.markdown("Enter the following details to predict diabetes:")

# Input fields for user data
pregnancies = st.number_input("Number of Pregnancies: ", min_value=0, max_value=10, value=1)
glucose = st.number_input("Glucose Level: ", min_value=0)
blood_pressure = st.number_input("Blood Pressure: ", min_value=0)
skin_thickness = st.number_input("Skin Thickness: ", min_value=0)
insulin = st.number_input("Insulin Level: ", min_value=0)
bmi = st.number_input("BMI: ", min_value=1)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function: ", min_value=0)
age = st.number_input("Age: ", min_value=0)

# Button to trigger prediction
if st.button("Predict Diabetes"):

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    
   
    input_scaled = scaler.transform(input_data)
    
   
    prediction = model.predict(input_scaled)
    
    
    result = "Not Diabetic" if prediction < 0.5 else "Diabetic"
   
    st.success(result)

=======
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

# load model and scaler
model = tf.keras.models.load_model("Diabetic_model.h5")
scaler = joblib.load("Scaler.pkl")

# App page
st.set_page_config(page_title="Diabetes Prediction App" , layout = "Centered")
st.title("Diabetes Prediction App")
st.markdown("enter the following details to predict the diabetes")

# Input field
pregnancies = st.number_input("Enter the number of pregnancies", min_value=0, max_value=10, step=1)
glucose = st.number_input("Enter the glucose level",min_value = 0)
blood_pressure = st.number_input("Enter the blood pressure",min_value = 0)
skinthickness = st.number_input("Enter the skin thickness",min_value=0)
insulin = st.number_input("Enter the amount of insulin in patients body",min_value= 0)
bmi = st.number_input("Enter the body weight of the patient : ",min_value = 0)
age = st.number_input("Enter the age of the patient : ",min_value = 0)

# make predictions
if st.button('Predict Diabetes'):
    # scale the input data
    input_data = scaler.transform([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetespedigreefunction, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0][0]
    result = "Not Diabetic " if prediction < 0.5 else "Diabetic"
    st.subheader("The result of the prediction are : ",result)
>>>>>>> cdcf8b828af9c14597699b265c893719dbc83166
