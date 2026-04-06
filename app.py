from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import os

app = Flask(__name__)

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/asr7code/ML-dataset/main/flight_dataset.csv')

# Encoding
le_airline = LabelEncoder()
le_source = LabelEncoder()
le_dest = LabelEncoder()

df["Airline"] = le_airline.fit_transform(df["Airline"])
df["Source"] = le_source.fit_transform(df["Source"])
df["Destination"] = le_dest.fit_transform(df["Destination"])

X = df.drop("Price", axis=1)
y = df["Price"]

model = RandomForestRegressor()
model.fit(X, y)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/form')
def form():
    return render_template("form.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        Airline = int(request.form['Airline'])
        Source = int(request.form['Source'])
        Destination = int(request.form['Destination'])
        Total_Stops = int(request.form['Total_Stops'])
        Date = int(request.form['Date'])

        # Auto-fill remaining features
        Month = 3
        Year = 2019
        Dep_hours = 10
        Dep_min = 0
        Arrival_hours = 12
        Arrival_min = 0
        Duration_hours = 2
        Duration_min = 30

        features = [[
            Airline, Source, Destination, Total_Stops,
            Date, Month, Year,
            Dep_hours, Dep_min,
            Arrival_hours, Arrival_min,
            Duration_hours, Duration_min
        ]]

        prediction = model.predict(features)[0]

        return render_template(
            "form.html",
            prediction_text=f"Estimated Price: ₹ {round(prediction, 2)}"
        )

    except Exception as e:
        return render_template("form.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
