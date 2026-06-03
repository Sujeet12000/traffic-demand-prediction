
import joblib
import math
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 1. Load Model and Encoders
model_data = joblib.load("models/rf_tuned_traffic.pkl")
model = model_data["model"]
feature_names = model_data["features"]
le_zone = model_data["le_zone"]
le_weather = model_data["le_weather"]
print("VALID ZONES:", le_zone.classes_)
print("VALID WEATHER:", le_weather.classes_)
# 2. Initialize FastAPI & CORS
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Request Schema (Matches React Frontend Payload)
class PredictionRequest(BaseModel):
    zone: str
    weather: str
    hour: int
    month: int
    holiday: int
    temperature: float

@app.get("/")
def home():
    return {"message": "Traffic AI Backend Running"}

# 4. The Engineering & Prediction Endpoint
@app.post("/predict")
def predict(data: PredictionRequest):
    try:
        # A. Label Encoding
        zone_enc = le_zone.transform([data.zone])[0]
        weather_enc = le_weather.transform([data.weather])[0]

        # B. Cyclical Time Encoding (Math)
        hour_sin = math.sin(2 * math.pi * data.hour / 24.0)
        hour_cos = math.cos(2 * math.pi * data.hour / 24.0)
        
        month_sin = math.sin(2 * math.pi * data.month / 12.0)
        month_cos = math.cos(2 * math.pi * data.month / 12.0)

        # C. Generate Missing Features (Day of Week & Peak Hour)
        # Assuming Wednesday (3) as a baseline since UI doesn't send exact date
        mock_dow = 3 
        dow_sin = math.sin(2 * math.pi * mock_dow / 7.0)
        dow_cos = math.cos(2 * math.pi * mock_dow / 7.0)

        # Define peak hours (e.g., 7 AM - 10 AM, 4 PM - 7 PM)
        is_peak = 1 if (7 <= data.hour <= 10) or (16 <= data.hour <= 19) else 0

        # D. Assemble the Exact 11-Feature Array
        features = [[
            hour_sin,       # 1
            hour_cos,       # 2
            dow_sin,        # 3
            dow_cos,        # 4
            month_sin,      # 5
            month_cos,      # 6
            is_peak,        # 7
            data.holiday,   # 8
            data.temperature,# 9
            zone_enc,       # 10
            weather_enc     # 11
        ]]

        # E. Predict
        prediction = model.predict(features)[0]

        return {
            "predicted_traffic": float(prediction) 
        }

    except Exception as e:
        print("ERROR:", e)
        return {"error": str(e)}