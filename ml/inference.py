import os
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel

# TODO: Import model when ready
from models import * # TODO: SPECIFY

class InputData(BaseModel):
    pass


weights_path = os.getenv("ML_LATEST_WEIGHTS_PATH")
if weights_path is None:
    raise RuntimeError("ML_LATEST_WEIGHTS_PATH not set")


# FastAPI app
app = FastAPI(title="ML Inference API", version="1.0.0")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "ml-inference"}

@app.post("/predict")
def predict(data: InputData):

    #TODO: x = torch.tensor([data.features], dtype=torch.float32)

    with torch.no_grad():

        #TODO: y = model(x)

        y=torch.tensor(0)
    return {"prediction": y.tolist()}
