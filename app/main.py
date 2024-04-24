from fastapi import FastAPI, Response
from pydantic import BaseModel
from prometheus_client import Summary, Counter, Histogram, Gauge
from prometheus_client.core import CollectorRegistry
from prometheus_client import make_asgi_app
import prometheus_client
from app.Model.model import Predict_pipeline
from app.Model.model import __version__ as ModelVersion
import time
from prometheus_fastapi_instrumentator import Instrumentator




app = FastAPI(debug=False)

_INF = float("inf")

graphs = {}
graphs['c'] = Counter('l_m_total_requests', 'Total numnber of language model requests' )
graphs['h'] = Histogram('l_m_request_latency_seconds', 'Histogram from Inference time', buckets=(1, 2, 5, 6, 10, _INF))

class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    language: str


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": ModelVersion}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    start = time.time()
    graphs['c'].inc()
    language = Predict_pipeline(payload.text)
    end = time.time()
    graphs['h'].observe(end - start)
    return {"language": language}

Instrumentator.instrument(app).expose(app)
