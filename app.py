import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import pandas as pd

# Load both models
with open('trained_model.pkl', 'rb') as rf_file:
    rf_model = pickle.load(rf_file)

with open('trained_xgb_model.pkl', 'rb') as xgb_file:
    xgb_model = pickle.load(xgb_file)


with open("bayesian_trace.pkl", "rb") as file:
    bayesian_trace = pickle.load(file)


mse = 207.15360107233352


def predict_price(model, user_inputs):

    columns = ['achievements', 'Accounting', 'Action', 'Adventure', 'Animation & Modeling', 'Audio Production', 'Casual',
               'Design & Illustration', 'Early Access', 'Education', 'Free to Play', 'Game Development', 'Gore',
               'Indie', 'Massively Multiplayer', 'Movie', 'Nudity', 'Photo Editing', 'RPG', 'Racing',
               'Sexual Content', 'Simulation', 'Software Training', 'Sports', 'Strategy', 'Utilities',
               'Video Production', 'Violent', 'Web Publishing', 'num_languages', 'positive_review_percentage']

    input_df = pd.DataFrame(columns=columns, index=[0]).fillna(0)

    for key, value in user_inputs.items():
        input_df[key][0] = value

    # Convert object columns to bool
    for col in input_df.columns:
        if input_df[col].dtype == 'object':
            input_df[col] = input_df[col].astype(bool)

    predicted_price = model.predict(input_df)

    return predicted_price


def predict_price_range(model, mse, user_inputs, confidence_interval=0.98):
    predicted_price = predict_price(model, user_inputs)[0]

    std_dev = np.sqrt(mse)
    margin_of_error = std_dev * \
        stats.t.ppf((1 + confidence_interval) / 2, 3162 - 1)

    lower_bound = predicted_price - margin_of_error
    upper_bound = predicted_price + margin_of_error

    return max(lower_bound, 0), upper_bound


st.title('Steam Game Price Prediction')

# Sidebar
st.sidebar.title('Select the game features')

# Add a dropdown menu to select a model
model_selection = st.sidebar.selectbox(
    'Select a model', options=['Random Forest', 'XGBoost'])

# Load the selected model
if model_selection == 'Random Forest':
    model = rf_model
else:
    model = xgb_model

achievements = st.sidebar.number_input(
    'Number of Achievements', min_value=0, value=0)
num_languages = st.sidebar.number_input(
    'Number of Languages', min_value=1, value=1)
positive_review_percentage = st.sidebar.number_input(
    'Positive Review Percentage', min_value=0, max_value=100, value=50)

tags = ['Accounting', 'Action', 'Adventure', 'Animation & Modeling', 'Audio Production', 'Casual', 'Design & Illustration', 'Early Access', 'Education', 'Free to Play', 'Game Development', 'Gore', 'Indie',
        'Massively Multiplayer', 'Movie', 'Nudity', 'Photo Editing', 'RPG', 'Racing', 'Sexual Content', 'Simulation', 'Software Training', 'Sports', 'Strategy', 'Utilities', 'Video Production', 'Violent', 'Web Publishing']
user_inputs = {}
for tag in tags:
    user_inputs[tag] = st.sidebar.checkbox(tag)

user_inputs['achievements'] = achievements
user_inputs['num_languages'] = num_languages
user_inputs['positive_review_percentage'] = positive_review_percentage

# Main
st.header('Predicted Price Range')

if st.button('Predict Price Range'):
    lower_bound, upper_bound = predict_price_range(
        model, mse, user_inputs, confidence_interval=0.95)
    st.write(f'Predicted price range: {lower_bound:.2f} to {upper_bound:.2f}')
