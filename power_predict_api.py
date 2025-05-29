from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd, joblib, os
from influxdb_client import InfluxDBClient
from prophet import Prophet
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

# ⬅️ .forecast_env 로부터 환경변수 불러오기
load_dotenv(dotenv_path=".forecast_env")

app = FastAPI(title="Power Forecast & Training API")

# Influx 설정 (환경변수 사용)
URL    = os.getenv("INFLUX_URL")
TOKEN  = os.getenv("INFLUX_TOKEN")
ORG    = os.getenv("INFLUX_ORG")
BUCKET = os.getenv("INFLUX_BUCKET")
MODEL_PATH = "models/power_model_global.pkl"

# 예측 응답 스키마
class ForecastResult(BaseModel):
    actual_kWh: float
    predicted_kWh: float
    estimated_monthly_kWh: float

# 1️⃣ 학습 엔드포인트
@app.post("/api/train")
def train_model():
    client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
    query = f'''
    from(bucket:"{BUCKET}")
      |> range(start:-90d)
      |> filter(fn:(r)=> r["_measurement"]=="power_watts")
      |> aggregateWindow(every:1d, fn:mean)
      |> keep(columns:["_time","_value"])
      |> sort(columns:["_time"])
    '''
    tables = client.query_api().query(query)
    records = []
    for t in tables:
        for r in t.records:
            v = r.get_value()
            if v is None: continue
            records.append({"ds": r.get_time(), "y": v * 24 / 1000})
    if not records:
        raise HTTPException(404, "학습 데이터가 없습니다.")
    df = pd.DataFrame(records)
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)

    model = Prophet()
    model.fit(df)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return {"status": "trained", "model_path": MODEL_PATH}


# 2️⃣ 예측 엔드포인트
@app.get("/api/forecast/monthly", response_model=ForecastResult)
def monthly_forecast():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(404, "모델이 없습니다. /api/train 를 먼저 호출하세요.")
    model = joblib.load(MODEL_PATH)

    today = pd.Timestamp.now(tz="UTC")
    start = today.replace(day=1).strftime("%Y-%m-%dT00:00:00Z")
    end_of_month = (today + relativedelta(day=31)).replace(hour=23, minute=59, second=59).tz_localize(None)

    client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
    query = f'''
    from(bucket:"{BUCKET}")
      |> range(start:{start})
      |> filter(fn:(r)=> r["_measurement"]=="power_watts")
      |> aggregateWindow(every:1d, fn:mean)
      |> keep(columns:["_time","_value"])
      |> sort(columns:["_time"])
    '''
    tables = client.query_api().query(query)
    records = []
    for t in tables:
        for r in t.records:
            v = r.get_value()
            if v is None: continue
            records.append({"ds": r.get_time(), "y": v * 24 / 1000})
    if not records:
        raise HTTPException(404, "예측할 데이터가 없습니다.")
    df = pd.DataFrame(records)
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)

    last = df["ds"].max()
    days_remain = (end_of_month - last).days
    future = model.make_future_dataframe(periods=days_remain, freq="D")
    forecast = model.predict(future)
    future_only = forecast[forecast["ds"] > last]

    actual = df["y"].sum()
    predicted = future_only["yhat"].sum()
    total = actual + predicted

    return ForecastResult(
        actual_kWh=round(actual, 2),
        predicted_kWh=round(predicted, 2),
        estimated_monthly_kWh=round(total, 2)
    )
