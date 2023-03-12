import pickle
from typing import List, Any

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.preprocessing_pipeline.pipeline import PreprocessingPipeline

preprocessing_pipeline: PreprocessingPipeline
model: Any

app: FastAPI = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.on_event("startup")
def on_start():
    try:
        global model
        model = pickle.load(open('binaries/best_model.pickle', 'rb'))
        global preprocessing_pipeline
        preprocessing_pipeline = pickle.load(open('binaries/preprocessing_pipeline.pickle', 'rb'))
    except Exception as e:
        print(e)
        exit(1)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df: pd.DataFrame = pd.json_normalize(item.dict())
    prediction: float = model.predict(preprocessing_pipeline.transform(df))
    return prediction


# по заданию параметр items: List[item], но тогда непонятно, как читать файл
@app.post("/predict_items")
def predict_items(file: UploadFile = File()) -> FileResponse:
    try:
        df: pd.DataFrame = pd.read_csv(file.file)
    except Exception:
        raise HTTPException(400, detail='Invalid file type (should be .csv)')

    df['selling_price'] = model.predict(preprocessing_pipeline.transform(df))
    df.to_csv('data/predicted_items.csv')
    return FileResponse('data/predicted_items.csv')
