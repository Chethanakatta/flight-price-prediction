# ✈️ Flight Price Prediction (End-to-End ML Project)

## 📌 Overview

This project predicts flight ticket prices using Machine Learning and provides an interactive UI using Streamlit.

## 🚀 Features

* Data preprocessing pipeline
* Random Forest Regression model
* End-to-end ML pipeline using sklearn
* Streamlit web app for prediction
* Real-time user input prediction

## 🛠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit
* Pickle

## 📂 Project Structure

```
├── app.py
├── train_model.py
├── model.pkl
├── flight.csv
├── requirements.txt
├── README.md
```

## ⚙️ Installation

```bash
git clone https://github.com/your-username/flight-price-prediction-end-to-end-ml.git
cd flight-price-prediction-end-to-end-ml
pip install -r requirements.txt
```

## ▶️ Run the App

```bash
streamlit run app.py
```

## 📊 Model Details

* Algorithm: Random Forest Regressor
* Evaluation Metric: R² Score

## 🎯 Input Features

* Airline
* Source & Destination
* Stops
* Duration
* Days Left
* Journey Date

## 💡 Output

Predicted Flight Price 💰

## 📌 Future Improvements

* Deploy on cloud (Render / AWS)
* Use advanced models (XGBoost)
* Add live flight data API

