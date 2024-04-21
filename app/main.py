from fastapi import FastAPI, Response
from pydantic import BaseModel
from prometheus_client import Summary, Counter, Histogram, Gauge
from prometheus_client.core import CollectorRegistry
import prometheus_client
from app.Model.model import Predict_pipeline
from app.Model.model import __version__ as ModelVersion
import time





app = FastAPI()

graphs = {}
graphs['c'] = Counter('l_m_total_requests', 'Total numnber of language model requests' )
graphs['h'] = Histogram('l_m_request_latency_seconds', 'Histogram from Inference time')

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
    graphs['s'].inc()
    language = Predict_pipeline(payload.text)
    end = time.time()
    graphs['h'].observe(end - start)
    return {"language": language}

@app.route("/metrics")
def requests_count():
    res = []
    for k,v in graphs.items():
        res.append(prometheus_client.generate_latest(v))
    return Response(res, mimetype="text/plain")

