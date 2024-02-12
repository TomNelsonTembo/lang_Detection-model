from fastapi import FastAPI
from pydantic import BaseModel
from app.Model.model import Predict_pipeline
from app.Model.model import __version__ as ModelVersion





app = FastAPI()


class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    language: str


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": ModelVersion}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    language = Predict_pipeline(payload.text)
    return {"language": language}