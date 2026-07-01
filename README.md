# 🪙 Bitcoin MLOps Forecast Pipeline

An automated, daily MLOps forecasting and data drift monitoring pipeline for Bitcoin USD market trends. This repository pulls real-time price and volume data, engineers technical indicators, trains a TensorFlow/Keras prediction neural network, evaluates data drift, and redeploys predictions.

🌐 **Live Streamlit App**: [bitcoin-mlops-pipeline-3xtnbgrrwezgp3rungbxip.streamlit.app](https://bitcoin-mlops-pipeline-3xtnbgrrwezgp3rungbxip.streamlit.app/)

---

## 🚀 Key Pipeline Features

*   **🔌 Live Data Ingestion**: Periodically fetches the last 90 days of daily Bitcoin closing price and volume indicators using the CoinGecko public API.
*   **🛠️ Feature Engineering**: Extracts 5 engineered technical metrics:
    *   `price_return` (daily percentage return)
    *   `sma_5` (5-day Simple Moving Average)
    *   `sma_10` (10-day Simple Moving Average)
    *   `volatility` (5-day price change volatility)
    *   `volume_change` (trading volume momentum)
*   **🧠 Deep Learning Trend Forecasting**: Trains a binary classifier Feedforward Neural Network using **TensorFlow/Keras** to forecast if tomorrow's price will rise (1) or fall (0).
*   **🚨 Data Drift & Self-Healing Loop**: If prediction validation accuracy falls below **65%**, the training script triggers automated retraining on the expanded historical dataset to realign model weights.
*   **📊 Telemetry Dashboard**: A dark obsidian & neon styled Streamlit interface displaying live prices, model parameters, validation curves, tomorrow's trend probability, and custom warning alert cards.
*   **⚙️ Automation**: Orchestrated using **GitHub Actions** daily Cron scheduling to query the API and rebuild weights.

---

## 📁 Repository Directory Structure

*   `.github/workflows/mlops_pipeline.yml` — Daily pipeline automation workflow.
*   `data/`
    *   `latest_ingested_data.csv` — Actively fetched daily dataset.
    *   `pipeline_metrics.json` — Latest execution manifest (accuracy, action taken).
    *   `historical_runs.json` — Execution history ledger.
*   `models/latest_model.keras` — Production TensorFlow model weights.
*   `src/`
    *   `utils.py` — Logging functions.
    *   `src/`
        *   `app.py` — Streamlit Telemetry Hub dashboard.
        *   `data_ingestion.py` — CoinGecko API client.
        *   `model_training.py` — Keras training, validation, and auto-retraining loop.
        *   `utils.py` — Helper functions.
*   `requirements.txt` — Project python dependencies.

---

## 🛠️ Local Setup & Run Instructions

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/austinajay/bitcoin-mlops-pipeline.git
    cd bitcoin-mlops-pipeline
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Execute components manually**:
    *   *Fetch daily data*:
        ```bash
        python src/src/data_ingestion.py
        ```
    *   *Train/Evaluate model*:
        ```bash
        python src/src/model_training.py
        ```
    *   *Start local dashboard*:
        ```bash
        streamlit run src/src/app.py
        ```
