import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Flight Price Prediction")

st.title("✈️ Flight Price Prediction (Regression Model)")

# -------------------------
# Load Model
# -------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

expected_columns = model.feature_names_in_

st.write("Enter Flight Details")

# -------------------------
# User Inputs
# -------------------------
Airline = st.selectbox("Airline", ["IndiGo", "Air India", "SpiceJet", "Vistara"])
Source = st.selectbox("Source", ["Delhi", "Mumbai", "Chennai", "Kolkata"])
Destination = st.selectbox("Destination", ["Delhi", "Mumbai", "Chennai", "Kolkata"])
Total_stops = st.selectbox("Total Stops", ["non-stop", "1 stop", "2 stops"])

Departure = st.text_input("Departure Time (Example: 10:00)")
Arrival = st.text_input("Arrival Time (Example: 12:30)")
Flight_code = st.text_input("Flight Code (Example: AI203)")
Class = st.selectbox("Class", ["Economy", "Business"])

Duration_in_hours = st.number_input("Duration (in hours)", min_value=1, max_value=24, value=2)
Days_left = st.number_input("Days Left for Journey", min_value=0, max_value=365, value=10)

Journey_day = st.slider("Journey Day", 1, 31, 10)
Journey_month = st.slider("Journey Month", 1, 12, 5)

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Price"):

    input_dict = {
        "Airline": Airline,
        "Source": Source,
        "Destination": Destination,
        "Total_stops": Total_stops,
        "Departure": Departure,
        "Arrival": Arrival,
        "Flight_code": Flight_code,
        "Class": Class,
        "Duration_in_hours": Duration_in_hours,
        "Days_left": Days_left,
        "Journey_day": Journey_day,
        "Journey_month": Journey_month
    }

    # Create dataframe
    input_data = pd.DataFrame([input_dict])

    # Reorder columns to match training
    input_data = input_data.reindex(columns=expected_columns)

    prediction = model.predict(input_data)

    st.success(f"Estimated Flight Price: ₹ {int(prediction[0])}")
