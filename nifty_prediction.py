import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib


# Load the trained Keras model
try:
    model = tf.keras.models.load_model("final_model.h5") # Replace 'your_model.h5' with your model's filename
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'your_model.h5' exists in the same directory.")
    st.stop()

# Load the MinMaxScaler (assuming you saved it) or initialize and fit on training data
try:
    import joblib
    scaler = joblib.load('scaler.joblib')  # Replace 'scaler.joblib' with your scaler's filename
except FileNotFoundError:
    st.error("Scaler file not found. Please make sure 'scaler.joblib' exists in the same directory.")
    st.stop()

# Define feature names
feature_names = ['Nifty Close Price', 'S&P 100 Close Price', 'Crude oil price',
                 'Nifty volatility index', 'USD to INR exchange rate', 'Gold Price', 'MACD', 'RSI']

# Streamlit UI
st.title('Price Prediction')

# Input fields for features
inputs = []
for feature in feature_names:
    user_input = st.number_input(f'Enter {feature}:')
    inputs.append(user_input)

# Prediction button
if st.button('Predict'):
    # Prepare input data
    input_data = np.array(inputs).reshape(1, -1)

    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)

    # Inverse transform the prediction to original scale
    predicted_price = scaler.inverse_transform(np.concatenate((scaled_input[:,:-1],prediction), axis=1))[:,-1] #assuming that the last column is the predicted column.

    st.write(f'Predicted Price: {predicted_price[0]:.2f}')