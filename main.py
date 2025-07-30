from fastapi import FastAPI
from pydantic import BaseModel 
import joblib
import numpy as np
import os
import csv
from datetime import datetime, timezone

# Load the trained model
model = joblib.load('model.joblib')

#Create FastAPI app
app = FastAPI()

#Define the input schema using pydantic
class HousingFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

#File paths
LOG_FILE = 'logs/predictions_log.csv'
CSV_FILE = 'data/predictions.csv'

#Ensure CSV file has headers
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                         'Population', 'AveOccup', 'Latitude', 'Longitude', 'predicted_price'])

#Define the prediction endpoint
@app.post("/predict")
def predict(features: HousingFeatures):
    #Convert input to array
    data = np.array([[getattr(features, field) for field in features.__annotations__.keys()]])

    #Predict
    prediction = model.predict(data)[0]
    price = round(float(prediction), 2)
    timestamp = datetime.now(timezone.utc).isoformat()

    #Log to file
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"{timestamp} Prediction: {price}\n")

    #Append to CSV
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            features.MedInc, features.HouseAge, features.AveRooms, features.AveBedrms,
            features.Population, features.AveOccup, features.Latitude, features.Longitude,
            price
        ])

    return {"predicted_price": price}