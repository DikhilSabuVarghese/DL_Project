import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

# Load the trained model
model = load_model(r"D:\DL Project\fifa_transfer_value_model.keras")

# Streamlit title and description
st.title("FIFA Player Transfer Value Prediction")
st.write("""
This app predicts the transfer value of FIFA players based on their performance attributes.
Select a player from the list or input their attributes to predict their transfer value.
""")

# Load player data (replace with your actual dataset or a subset of it)
df = pd.read_csv(r"D:\DL Project\female_players.csv")  # Ensure this file is correctly referenced
player_names = df['short_name'].unique()  # Assuming 'short_name' column contains player names

# Add a dropdown for selecting a player name
player_name = st.selectbox("Select Player", player_names)

# Display selected player's details
player_data = df[df['short_name'] == player_name].iloc[0]

st.write(f"### {player_name}")
st.write(f"Overall: {player_data['overall']}")
st.write(f"Potential: {player_data['potential']}")
st.write(f"Pace: {player_data['pace']}")
st.write(f"Shooting: {player_data['shooting']}")
st.write(f"Dribbling: {player_data['dribbling']}")

# Get player attributes from the selected player (or slider inputs)
overall = player_data['overall']
potential = player_data['potential']
pace = player_data['pace']
shooting = player_data['shooting']
dribbling = player_data['dribbling']

# Prepare the input sequence for LSTM prediction (Assuming LSTM requires 5 time steps and 5 features)
sequence_length = 5
features = [overall, potential, pace, shooting, dribbling]

# Create the sequence for LSTM (expand the input to match the sequence length)
sequence = np.tile(features, (sequence_length, 1)).reshape(1, sequence_length, len(features))

# Predict and display result when the button is clicked
if st.button("Predict Transfer Value"):
    try:
        # Predict the transfer value
        prediction = model.predict(sequence)
        predicted_value = np.expm1(prediction[0][0])  # Reverse the log transformation (if used)
        st.write(f"### Predicted Transfer Value for {player_name}: €{predicted_value:,.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
