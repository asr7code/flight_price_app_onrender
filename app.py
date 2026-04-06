from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import os

app = Flask(__name__)

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/asr7code/ML-dataset/main/flight_dataset.csv')

# Encode
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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [int(x) for x in request.form.values()]
        prediction = model.predict([data])[0]
        return render_template("index.html", prediction_text=f"Estimated Price: ₹ {round(prediction,2)}")
    except Exception as e:
        return render_template("index.html", prediction_text=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
