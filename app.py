from fastapi import FastAPI
import pandas as pd 
from src.pipeline.predict_pipeline import PredictPipeline
from pydantic import BaseModel
app = FastAPI()
pipeline = PredictPipeline()

class ChurnInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tensure: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity:str
    OnlineBackup:str
    DeviceProtection:str
    TechSupport:str
    StreamingTV:str
    StreamingMovies:str
    Contract:str
    PaperlessBilling:str
    PaymentMethod:str
    MonthlyCharges:float
    TotalCharges:float

@app.post("/predict")
def predict_churn(data:ChurnInput):
    df = pd.DataFrame([data.dict()])
    preds = pipeline.predict(df)
    return{"churn":"Yes" if preds[0] == 1 else "No"}

