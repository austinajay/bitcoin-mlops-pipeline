import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sklearn.preprocessing import StandardScaler

# Locate directories relative to this file
SRC_SRC_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SRC_SRC_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PYTHON_EXE = sys.executable

app = FastAPI(title="Bitcoin MLOps API Server")

# Allow CORS for static site hosting integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/api/metrics")
def get_metrics():
    metrics_path = os.path.join(DATA_DIR, "pipeline_metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=444, detail="Pipeline metrics not initialized yet.")
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading metrics: {str(e)}")

@app.get("/api/history")
def get_history():
    history_path = os.path.join(DATA_DIR, "historical_runs.json")
    if not os.path.exists(history_path):
        return []
    try:
        with open(history_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading history: {str(e)}")

@app.get("/api/data")
def get_data():
    data_path = os.path.join(DATA_DIR, "latest_ingested_data.csv")
    if not os.path.exists(data_path):
        raise HTTPException(status_code=444, detail="Dataset not ingested yet.")
    try:
        df = pd.read_csv(data_path)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing dataset: {str(e)}")

@app.get("/api/predict")
def get_prediction():
    model_file = os.path.join(MODELS_DIR, "latest_model.keras")
    data_path = os.path.join(DATA_DIR, "latest_ingested_data.csv")
    
    if not os.path.exists(model_file) or not os.path.exists(data_path):
        raise HTTPException(status_code=444, detail="Model weights or dataset not initialized.")
        
    try:
        # Load model locally
        import tensorflow as tf
        model = tf.keras.models.load_model(model_file)
        
        # Load data
        df = pd.read_csv(data_path)
        FEATURES = ["price", "volume", "price_return", "sma_5", "sma_10", "volatility", "volume_change"]
        X = df[FEATURES].values
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Run inference
        preds = model.predict(X_scaled, verbose=0).flatten()
        
        # Format results
        output = []
        for i in range(len(df)):
            output.append({
                "date": df.iloc[i]["date"],
                "price": float(df.iloc[i]["price"]),
                "volume": float(df.iloc[i]["volume"]),
                "target": int(df.iloc[i]["target"]),
                "probability": float(preds[i]),
                "prediction_class": int(preds[i] > 0.5)
            })
            
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/api/run-ingestion")
def trigger_ingestion():
    script_path = os.path.join(SRC_SRC_DIR, "data_ingestion.py")
    try:
        res = subprocess.run([PYTHON_EXE, script_path], capture_output=True, text=True, cwd=PROJECT_ROOT)
        return {
            "status": "success" if res.returncode == 0 else "failed",
            "exit_code": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute ingestion script: {str(e)}")

@app.post("/api/run-training")
def trigger_training():
    script_path = os.path.join(SRC_SRC_DIR, "model_training.py")
    try:
        # Delete old model first to trigger fresh base training if model doesn't exist
        res = subprocess.run([PYTHON_EXE, script_path], capture_output=True, text=True, cwd=PROJECT_ROOT)
        return {
            "status": "success" if res.returncode == 0 else "failed",
            "exit_code": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute training script: {str(e)}")

@app.post("/api/simulate-drift")
def simulate_drift():
    data_path = os.path.join(DATA_DIR, "latest_ingested_data.csv")
    script_path = os.path.join(SRC_SRC_DIR, "model_training.py")
    if not os.path.exists(data_path):
        raise HTTPException(status_code=444, detail="Ingested data file not found.")
    try:
        # Invert target labels to simulate model drift
        df = pd.read_csv(data_path)
        df["target"] = 1 - df["target"]
        df.to_csv(data_path, index=False)
        
        # Trigger retraining cycle
        res = subprocess.run([PYTHON_EXE, script_path], capture_output=True, text=True, cwd=PROJECT_ROOT)
        return {
            "status": "success" if res.returncode == 0 else "failed",
            "exit_code": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift simulation failed: {str(e)}")

# Mount static files folder
static_dir = os.path.join(SRC_SRC_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    # Start web server on port 8000
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
