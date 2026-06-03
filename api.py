
import joblib

model_data = joblib.load("models/rf_tuned_traffic.pkl")

model = model_data["model"]
feature_names = model_data["features"]
le_zone = model_data["le_zone"]
le_weather = model_data["le_weather"]

print(feature_names)
print(le_zone.classes_)
print(le_weather.classes_)
print(model_data.keys())



from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Traffic AI Backend Running"}
@app.get("/test")
def test():
    return {"status": "working"}


from pydantic import BaseModel
class PredictionRequest(BaseModel):
    zone: str
    weather: str


@app.post("/predict")
def predict(data: PredictionRequest):
    try:
        zone_encoded = le_zone.transform([data.zone])[0]
        weather_encoded = le_weather.transform([data.weather])[0]

        features = [[
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            25,
            zone_encoded,
            weather_encoded
        ]]

        print(features)

        prediction = model.predict(features)[0]

        return {
            "prediction": float(prediction)
        }

    except Exception as e:
        print("ERROR:", e)
        return {"error": str(e)}