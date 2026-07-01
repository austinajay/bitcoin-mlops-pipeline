import os
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from utils import log_daily_run

MODEL_PATH = "models/latest_model.keras"

def build_simple_nn(input_dim):
    """Constructs a basic neural network for binary trend detection."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def run_pipeline_cycle():
    if not os.path.exists("data/latest_ingested_data.csv"):
        print("No fresh data found. Exiting.")
        return

    df = pd.read_csv("data/latest_ingested_data.csv")
    FEATURES = ["price", "volume", "price_return", "sma_5", "sma_10", "volatility", "volume_change"]
    X = df[FEATURES].values
    y = df["target"].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    current_date = datetime.utcnow().strftime("%Y-%m-%d")
    action_taken = "Evaluated"
    
    # 1. Check if model exists; if not, initialize and train it
    if not os.path.exists(MODEL_PATH):
        print("No pre-existing model found. Training base model...")
        model = build_simple_nn(X_scaled.shape[1])
        model.fit(X_scaled, y, epochs=30, batch_size=4, verbose=0)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save(MODEL_PATH)
        action_taken = "Initialized & Trained"
        accuracy = float(model.evaluate(X_scaled, y, verbose=0)[1])
    else:
        # 2. Evaluate existing model on fresh data
        model = tf.keras.models.load_model(MODEL_PATH)
        loss, accuracy = model.evaluate(X_scaled, y, verbose=0)
        accuracy = float(accuracy)
        
        # 3. Handle Data/Performance Drift Trigger
        # If accuracy drops below 65%, trigger an automated retraining cycle
        if accuracy < 0.65:
            print(f"Performance drift detected (Accuracy: {accuracy:.2f}). Retraining model...")
            model.fit(X_scaled, y, epochs=40, batch_size=4, verbose=0)
            model.save(MODEL_PATH)
            action_taken = "Retrained due to Drift"
            loss, accuracy = model.evaluate(X_scaled, y, verbose=0)
            accuracy = float(accuracy)

    # 4. Log metrics to JSON for the Streamlit UI to display
    latest_price = float(df["price"].iloc[-1])
    latest_volume = float(df["volume"].iloc[-1])
    
    metrics = {
        "date": current_date,
        "latest_price": latest_price,
        "latest_volume": latest_volume,
        "model_accuracy": round(accuracy, 4),
        "action_taken": action_taken
    }
    
    log_daily_run(metrics)
    print(f"Pipeline executed successfully. Action: {action_taken}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    run_pipeline_cycle()