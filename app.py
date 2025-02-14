import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.h5')

with open('onehotencoder.pkl', 'rb') as file:
    onehotencoder = pickle.load(file)
    
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
    
    
st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', onehotencoder.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100, 30)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_credit_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'])

has_credit_card = 1 if has_credit_card == 'Yes' else 0
is_active_member = 1 if is_active_member == 'Yes' else 0

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

geo_encoded = onehotencoder.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehotencoder.get_feature_names_out(['Geography']))

input_df = pd.concat([input_data, geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_df)

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]


st.write('Prediction Probability:', prediction_proba)

if (prediction_proba > 0.5):
    st.write('Customer is likely to churn')
else:
    st.write('Customer is not likely to churn') 