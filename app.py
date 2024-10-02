import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pandas as pd
import pickle

model = tf.keras.models.load_model('D:\Deep_Learning_Projects\Churn_prediction_ANN\chrun_classifier.h5')

with open('D:\Deep_Learning_Projects\Churn_prediction_ANN\gender_label_encoder.pkl', 'rb') as file:
    gender_encoder = pickle.load(file)

print(gender_encoder.classes_)

with open('D:\Deep_Learning_Projects\Churn_prediction_ANN\geography_one_hot_encoder.pkl', 'rb') as file:
    geography_encoder = pickle.load(file)

with open('D:\Deep_Learning_Projects\Churn_prediction_ANN\scaler.pkl', 'rb') as file:
    scaler_model = pickle.load(file)



st.title("Welcome to Customer Chrun Prediction App!")

geography = st.selectbox('Enter the Geography', geography_encoder.categories_[0])
gender = st.selectbox("Enter the Gender", gender_encoder.classes_)
age = st.slider('Enter the Age', 18, 90)
balance = st.number_input("Enter the balance")
credit_score = st.number_input("Enter the credit score")
estimated_salary = st.number_input("Enter the estimated salay")
tenure = st.slider("Enter the tenure", 1, 10)
num_of_products = st.slider("Enter the number of products", 1, 5)
has_credit_card = st.selectbox("Do you have credit card?", [0, 1])
is_active_member = st.selectbox("Are you an active member", [0,1])

input_data = pd.DataFrame(
    {
        'CreditScore': [credit_score],
        'Gender': [gender_encoder.transform([gender])[0]],
        'Age': [age], 
        'Tenure': [tenure], 
        'Balance': [balance], 
        'NumOfProducts': [num_of_products], 
        'HasCrCard': [has_credit_card],
       'IsActiveMember': [is_active_member], 
       'EstimatedSalary': [estimated_salary] 
    }
)


input_geo_encoded = geography_encoder.transform([[geography]]).toarray()
input_geo_encoded_df = pd.DataFrame(input_geo_encoded, columns=geography_encoder.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), input_geo_encoded_df], axis=1)
input_data = scaler_model.transform(input_data)
prediction = model.predict(input_data)
prediction_prob = prediction[0][0]

st.write(f"The prediction probability is: {prediction[0][0]}")

if prediction_prob > 0.5:
    st.write("Oof! The customer is likey to chrun.")

else:
    st.write("Horray! the customer is not likely to chrun.")