import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load model
model = pickle.load(open("cricket_match_predictor.pkl", "rb"))

winner_encoder = pickle.load(open("winner_encoder.pkl","rb"))
team1_encoder = pickle.load(open("team1_encoder.pkl","rb"))
team2_encoder = pickle.load(open("team2_encoder.pkl","rb"))
ground_encoder = pickle.load(open("ground_encoder.pkl","rb"))

# Load dataset to get team and ground names
df = pd.read_csv("dataset/ODI_cricinfo.csv")

teams = sorted(list(set(df["Team 1"]).union(set(df["Team 2"]))))
grounds = sorted(df["Ground"].unique())

#Label encoder
le = LabelEncoder()
le.fit(teams + grounds)

st.title("Cricket Match Winner Prediction")

team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", teams)
ground = st.selectbox("Select Ground", grounds)

if st.button("Predict Winner"):

    team1_encoded = le.transform([team1])[0]
    team2_encoded = le.transform([team2])[0]
    ground_encoded = le.transform([ground])[0]

    prediction = model.predict([[team1_encoded, team2_encoded, ground_encoded]])

    winner = le.inverse_transform(prediction)[0]

    st.success(f"Predicted Winner: {winner}")