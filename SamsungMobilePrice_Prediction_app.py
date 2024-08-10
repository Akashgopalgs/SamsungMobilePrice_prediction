import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model, encoder, and scaler
model = joblib.load('best_model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Extract known categories
known_models = encoder.categories_[0]
known_displays = encoder.categories_[1]
known_colours = encoder.categories_[2]
known_back_cameras = encoder.categories_[3]
known_front_cameras = encoder.categories_[4]

# Streamlit App
st.title('Samsung Mobile Price Prediction')

# Input fields
model_name = st.selectbox('Model', known_models)
display = st.selectbox('Display', known_displays)
colour = st.selectbox('Colour', known_colours)
back_camera = st.selectbox('Back Camera', known_back_cameras)
front_camera = st.selectbox('Front Camera', known_front_cameras)

# Using selectbox instead of number_input for battery, storage, and ram
battery = st.selectbox('Battery Capacity (mAh)', options=[3000, 3500, 4000, 4500, 5000, 5500, 6000])
storage = st.selectbox('Storage (GB)', options=[32, 64, 128, 256, 512])
ram = st.selectbox('RAM (GB)', options=[2, 4, 6, 8, 12, 16])

rating = st.slider('Rating', 0.0, 5.0, 4.2)

# Encode categorical inputs
input_data = pd.DataFrame([[model_name, display, colour, back_camera, front_camera, rating, battery, storage, ram]],
                          columns=['model', 'Display', 'colour', 'back camera', 'front camera', 'Rating', 'Battery', 'storage', 'ram'])

encoded_input = encoder.transform(input_data[['model', 'Display', 'colour', 'back camera', 'front camera']])
encoded_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(['model', 'Display', 'colour', 'back camera', 'front camera']))

# Scale numerical inputs
scaled_input = scaler.transform(input_data[['Rating', 'Battery', 'storage', 'ram']])

# Combine encoded and scaled inputs
final_input = np.hstack([scaled_input, encoded_df])

# Predict and display the result
if st.button('Predict Price'):
    prediction = model.predict(final_input)
    st.success(f'#### Estimated Price: ₹{prediction[0]:.2f}')
