import os
import sys
import json
import time
import subprocess
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Handle TensorFlow warning suppression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Directory configuration relative to this file
SRC_SRC_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SRC_SRC_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PYTHON_EXE = sys.executable

st.set_page_config(
    page_title="Bitcoin MLOps Forecast Telemetry",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Obsidian-neon glassmorphism styling
st.markdown("""
<style>
    /* Dark Obsidian Palette */
    .stApp {
        background-color: #0b0c10;
        color: #c5c6c7;
    }
    
    /* Elegant Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    
    /* Neon Glow metrics */
    div[data-testid="stMetricValue"] {
        font-size: 2.3rem !important;
        font-weight: 800 !important;
        color: #66fcf1 !important;
    }
    
    div[data-testid="metric-container"] {
        background: rgba(31, 40, 51, 0.45);
        border: 1px solid rgba(102, 252, 241, 0.15);
        border-radius: 12px;
        padding: 18px 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        border-color: rgba(102, 252, 241, 0.4);
        box-shadow: 0 4px 30px rgba(102, 252, 241, 0.1);
        transform: translateY(-2px);
    }
    
    /* Standard alerts overriding default colors to fit the theme */
    .stAlert {
        background: rgba(31, 40, 51, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        color: #c5c6c7 !important;
    }
    
    /* Styled Neon Button */
    .stButton>button {
        background: linear-gradient(135deg, #1f2833 0%, #45f3ff 200%) !important;
        color: #ffffff !important;
        border: 1px solid #66fcf1 !important;
        border-radius: 8px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 4px 15px rgba(102, 252, 241, 0.1) !important;
    }
    
    .stButton>button:hover {
        background: #66fcf1 !important;
        color: #0b0c10 !important;
        box-shadow: 0 0 20px rgba(102, 252, 241, 0.6) !important;
        transform: scale(1.02) !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to run scripts and capture output
def run_pipeline_component(script_name, env_vars=None):
    script_path = os.path.join(SRC_SRC_DIR, script_name)
    cmd = [PYTHON_EXE, script_path]
    
    current_env = os.environ.copy()
    if env_vars:
        current_env.update(env_vars)
        
    with st.spinner(f"Executing pipeline task: {script_name}..."):
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, env=current_env)
    return result

# Load Metrics & Data Helpers
def load_latest_metrics():
    metrics_path = os.path.join(DATA_DIR, "pipeline_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def load_run_history():
    history_path = os.path.join(DATA_DIR, "historical_runs.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def load_datasets():
    data_path = os.path.join(DATA_DIR, "latest_ingested_data.csv")
    if os.path.exists(data_path):
        try:
            return pd.read_csv(data_path)
        except Exception:
            pass
    return None

# Load model and compile if necessary
def load_keras_model():
    model_file = os.path.join(MODELS_DIR, "latest_model.keras")
    if os.path.exists(model_file):
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(model_file)
        except Exception as e:
            st.error(f"Error loading keras model: {e}")
    return None

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.markdown("<h2 style='text-align: center; color: #66fcf1;'>⚙️ MLOps Controls</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.subheader("1. Ingestion Config")
st.sidebar.markdown("**Endpoint:** CoinGecko API")
st.sidebar.markdown("**Asset:** Bitcoin (BTC)")
st.sidebar.markdown("**Historical Window:** 30 Days")

st.sidebar.subheader("2. Target Setting")
st.sidebar.write("Predict if Bitcoin price goes up (1) or down (0) tomorrow based on daily price and volume fluctuations.")

st.sidebar.markdown("---")
st.sidebar.subheader("3. Interactive Triggers")
trigger_ingestion = st.sidebar.button("Fetch Market Data")
trigger_evaluation = st.sidebar.button("Run Evaluation Cycle")

st.sidebar.markdown("---")
# Drift simulation help
st.sidebar.subheader("4. Inject Drift Simulator")
st.sidebar.write("Force a performance drop (accuracy < 65%) to test the automatic retraining cycle of the pipeline.")
simulate_drift = st.sidebar.button("Simulate Drift Event")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align: center; color: #45f3ff; font-size: 0.8rem;'>MLOps Daily Pipeline<br>V1.2.0 • TensorFlow Keras</div>",
    unsafe_allow_html=True
)

# ----------------- ACTIONS HANDLERS -----------------
if trigger_ingestion:
    res = run_pipeline_component("data_ingestion.py")
    if res.returncode == 0:
        st.success("Data Ingestion complete! Pulled latest CoinGecko Bitcoin market values.")
        st.rerun()
    else:
        st.error(f"Data Ingestion failed! Log: {res.stderr}")

if trigger_evaluation:
    res = run_pipeline_component("model_training.py")
    if res.returncode == 0:
        st.success("Evaluation cycle finished successfully.")
        st.rerun()
    else:
        st.error(f"Evaluation cycle failed! Log: {res.stderr}")

if simulate_drift:
    # We can simulate drift by injecting a file with inverted targets so the model gets low accuracy
    data_path = os.path.join(DATA_DIR, "latest_ingested_data.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        # Invert target values to destroy predictive accuracy
        df["target"] = 1 - df["target"]
        df.to_csv(data_path, index=False)
        st.info("Simulated drift data injected. Running evaluation loop to trigger retraining...")
        res = run_pipeline_component("model_training.py")
        if res.returncode == 0:
            st.success("Retrained successfully! Check the status alert below.")
            st.rerun()
        else:
            st.error(f"Drift retraining failed! Log: {res.stderr}")
    else:
        st.warning("Please bootstrap the pipeline first to generate data.")

# ----------------- MAIN CORE -----------------
# Header banner
st.markdown("""
<div style="background: linear-gradient(90deg, #1f2833 0%, #0b0c10 100%); padding: 25px; border-radius: 14px; border: 1px solid rgba(102, 252, 241, 0.15); margin-bottom: 25px; box-shadow: 0 4px 30px rgba(0,0,0,0.5);">
    <h1 style="margin: 0; font-size: 2.4rem; background: linear-gradient(to right, #66fcf1, #45f3ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        🪙 Bitcoin MLOps Daily Telemetry Hub
    </h1>
    <p style="margin: 5px 0 0 0; color: #c5c6c7; font-size: 1.1rem;">
        Automated data drift mitigation, Keras Neural Net model health auditing, and daily training scheduling.
    </p>
</div>
""", unsafe_allow_html=True)

# Load pipeline files
latest_metrics = load_latest_metrics()
run_history = load_run_history()
df_data = load_datasets()

# Pipeline bootstrapper if files missing
if df_data is None or latest_metrics is None:
    st.warning("⚠️ No operational pipeline artifacts detected. Run initial bootstrap to generate dataset and train the TensorFlow network.")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        bootstrap_btn = st.button("🚀 Initialize & Train Base Pipeline")
        if bootstrap_btn:
            st.write("Fetching historical CoinGecko prices...")
            r1 = run_pipeline_component("data_ingestion.py")
            st.write("Initializing Keras model & starting training...")
            r2 = run_pipeline_component("model_training.py")
            if r1.returncode == 0 and r2.returncode == 0:
                st.success("Bootstrap completed successfully!")
                st.rerun()
            else:
                st.error("Bootstrap execution failed. Verify network connectivity.")
    st.stop()

# ----------------- LIVE METRICS PANEL -----------------
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

price_usd = latest_metrics.get("latest_price", 0.0)
volume_usd = latest_metrics.get("latest_volume", 0.0)
accuracy = latest_metrics.get("model_accuracy", 0.0)
action = latest_metrics.get("action_taken", "Evaluated")

# Status badges
status_val = "STABLE"
status_delta = "No Drift"
if "Retrained" in action:
    status_val = "RETRAINED"
    status_delta = "Drift Resolved"
elif "Initialized" in action:
    status_val = "INITIALIZED"
    status_delta = "Fresh Model"

with col_m1:
    st.metric(label="🪙 Bitcoin price (USD)", value=f"${price_usd:,.2f}")
with col_m2:
    st.metric(label="📊 24h Trading Volume", value=f"${volume_usd:,.0f}")
with col_m3:
    st.metric(label="🎯 Current Validation Accuracy", value=f"{accuracy:.2%}", delta=f"{accuracy - 0.65:.2%} above threshold")
with col_m4:
    st.metric(label="📈 Pipeline Health Status", value=status_val, delta=status_delta)

# Health state messaging
if "Retrained" in action:
    st.markdown(f"""
    <div style="background: rgba(255, 75, 75, 0.08); border: 1px solid rgba(255, 75, 75, 0.3); border-radius: 12px; padding: 18px; margin: 15px 0; box-shadow: 0 4px 20px rgba(255, 75, 75, 0.1); display: flex; align-items: center;">
        <div style="font-size: 26px; margin-right: 15px; filter: drop-shadow(0 0 5px rgba(255, 75, 75, 0.4));">🚨</div>
        <div>
            <h4 style="margin: 0; color: #ff4b4b; font-size: 1.1rem; font-weight: 700;">Auto-Retrain Event Triggered</h4>
            <p style="margin: 4px 0 0 0; color: #c5c6c7; font-size: 0.95rem; line-height: 1.45;">
                Model validation accuracy dropped below <b>65%</b> due to data drift. The pipeline completed an automated training run on updated data at <b>{latest_metrics.get('date')}</b>, saving updated weights to production.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
elif accuracy < 0.65:
    st.markdown("""
    <div style="background: rgba(255, 170, 0, 0.08); border: 1px solid rgba(255, 170, 0, 0.3); border-radius: 12px; padding: 18px; margin: 15px 0; box-shadow: 0 4px 20px rgba(255, 170, 0, 0.1); display: flex; align-items: center;">
        <div style="font-size: 26px; margin-right: 15px; filter: drop-shadow(0 0 5px rgba(255, 170, 0, 0.4));">⚠️</div>
        <div>
            <h4 style="margin: 0; color: #ffaa00; font-size: 1.1rem; font-weight: 700;">Performance Margin Warning</h4>
            <p style="margin: 4px 0 0 0; color: #c5c6c7; font-size: 0.95rem; line-height: 1.45;">
                Validation accuracy has dropped close to the retraining trigger threshold. High probability of incoming data drift.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="background: rgba(102, 252, 241, 0.06); border: 1px solid rgba(102, 252, 241, 0.25); border-radius: 12px; padding: 18px; margin: 15px 0; box-shadow: 0 4px 20px rgba(102, 252, 241, 0.08); display: flex; align-items: center;">
        <div style="font-size: 26px; margin-right: 15px; filter: drop-shadow(0 0 5px rgba(102, 252, 241, 0.4));">🟢</div>
        <div>
            <h4 style="margin: 0; color: #66fcf1; font-size: 1.1rem; font-weight: 700;">Pipeline Status Stable</h4>
            <p style="margin: 4px 0 0 0; color: #c5c6c7; font-size: 0.95rem; line-height: 1.45;">
                All systems operational. Validation accuracy is safely above the 65% limit. Automatic daily cron checks are passing.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------- DYNAMIC TELEMETRY CHARTS -----------------
st.markdown("### 📈 Ingestion & Pipeline Trends")
col_c1, col_c2 = st.columns([3, 2])

with col_c1:
    st.markdown("#### Bitcoin 30-Day Market Volume & Price Movement")
    df_data["Formatted Date"] = pd.to_datetime(df_data["date"])
    
    fig = go.Figure()
    # Volume Bars
    fig.add_trace(go.Bar(
        x=df_data["date"],
        y=df_data["volume"],
        name="Trading Volume (USD)",
        yaxis="y2",
        marker_color="rgba(102, 252, 241, 0.25)"
    ))
    # Price line
    fig.add_trace(go.Scatter(
        x=df_data["date"],
        y=df_data["price"],
        name="Close Price (USD)",
        mode="lines+markers",
        line=dict(color="#66fcf1", width=3),
        marker=dict(size=6, color="#45f3ff")
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Timeline",
        yaxis=dict(title=dict(text="Price (USD)", font=dict(color="#66fcf1")), tickfont=dict(color="#66fcf1")),
        yaxis2=dict(
            title=dict(text="Volume (USD)", font=dict(color="#45f3ff")), 
            tickfont=dict(color="#45f3ff"),
            overlaying="y", 
            side="right"
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_c2:
    st.markdown("#### Pipeline Accuracy History & Retrain Target")
    
    if len(run_history) > 0:
        history_df = pd.DataFrame(run_history)
        
        fig2 = go.Figure()
        # Accuracy line
        fig2.add_trace(go.Scatter(
            x=history_df["date"],
            y=history_df["model_accuracy"],
            mode="lines+markers",
            name="Run Accuracy",
            line=dict(color="#c5c6c7", width=2),
            marker=dict(
                size=8,
                color=history_df["action_taken"].apply(lambda a: "#FF3F3F" if "Retrained" in a else "#66fcf1")
            )
        ))
        # Threshold line
        fig2.add_trace(go.Scatter(
            x=[history_df["date"].min(), history_df["date"].max()],
            y=[0.65, 0.65],
            mode="lines",
            name="Drift Limit (65%)",
            line=dict(color="#FF3F3F", width=2, dash="dash")
        ))
        
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Date",
            yaxis_title="Validation Accuracy",
            yaxis=dict(range=[0.0, 1.05]),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Run evaluation multiple times to visualize historical trend accuracy.")

# ----------------- DETAILED ANALYSIS TAB PANELS -----------------
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 Ingested Data Grid", "🔮 Live Inference Engine", "⚙️ MLOps System Diagnostics"])

with tab1:
    st.markdown("#### Bitcoin Ingested Window (Used for evaluation/retraining)")
    display_df = df_data.copy().reset_index()
    display_df["Trend Target"] = display_df["target"].apply(lambda t: "🟢 Price Up" if t == 1 else "🔴 Price Down")
    display_df = display_df.sort_values(by="date", ascending=False)
    
    st.dataframe(
        display_df[["date", "price", "volume", "Trend Target"]],
        use_container_width=True,
        column_config={
            "date": "Date",
            "price": st.column_config.NumberColumn("Price (USD)", format="$%,.2f"),
            "volume": st.column_config.NumberColumn("Volume (USD)", format="$%,.0f"),
            "Trend Target": "Tomorrow's Actual Trend"
        }
    )

with tab2:
    st.markdown("#### Live Predictions & Tomorrow's Bitcoin Forecast")
    
    # Load model and run inference
    model = load_keras_model()
    if model is not None and df_data is not None:
        try:
            FEATURES = ["price", "volume", "price_return", "sma_5", "sma_10", "volatility", "volume_change"]
            X = df_data[FEATURES].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Predict
            preds = model.predict(X_scaled, verbose=0).flatten()
            df_preds = df_data.copy()
            df_preds["probability"] = preds
            df_preds["prediction_class"] = (preds > 0.5).astype(int)
            
            # Calculate tomorrow's forecast
            latest_row = df_preds.iloc[-1]
            prob = latest_row["probability"]
            forecast_class = "UP TREND" if prob > 0.5 else "DOWN TREND"
            forecast_color = "#00FFCC" if prob > 0.5 else "#FF3F3F"
            prob_percent = prob if prob > 0.5 else (1.0 - prob)
            
            col_inf1, col_inf2 = st.columns([1, 2])
            with col_inf1:
                st.markdown(f"""
                <div style="background: rgba(31, 40, 51, 0.45); border: 2px solid {forecast_color}; border-radius: 12px; padding: 25px; text-align: center; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.45);">
                    <h3 style="margin: 0; color: #c5c6c7;">Tomorrow's Forecast</h3>
                    <h1 style="margin: 15px 0; font-size: 2.2rem; color: {forecast_color} !important;">{forecast_class}</h1>
                    <p style="margin: 0; font-size: 1.1rem; color: #ffffff;">Confidence: <b>{prob_percent:.2%}</b></p>
                </div>
                """, unsafe_allow_html=True)
                
            with col_inf2:
                st.markdown("#### Forecast Sequence History (Last 10 Days)")
                df_preds["Inferred Tomorrow's Trend"] = df_preds["prediction_class"].apply(lambda c: "🟢 Up" if c == 1 else "🔴 Down")
                df_preds["Confidence Level"] = df_preds["probability"].apply(lambda p: f"{p:.2%}" if p > 0.5 else f"{1-p:.2%}")
                df_preds["Actual Tomorrow's Trend"] = df_preds["target"].apply(lambda t: "🟢 Up" if t == 1 else "🔴 Down")
                
                # Highlight correctness
                correctness = []
                for idx, r in df_preds.iterrows():
                    correctness.append("✅ Correct" if r["prediction_class"] == r["target"] else "❌ Incorrect")
                df_preds["Validation Outcome"] = correctness
                
                grid_preds = df_preds.sort_values(by="date", ascending=False)
                st.dataframe(
                    grid_preds[["date", "price", "Inferred Tomorrow's Trend", "Confidence Level", "Actual Tomorrow's Trend", "Validation Outcome"]],
                    use_container_width=True,
                    column_config={
                        "price": st.column_config.NumberColumn("Price (USD)", format="$%,.2f")
                    }
                )
        except Exception as e:
            st.error(f"Inference run failed: {e}")
    else:
        st.info("Keras Model not found. Train the model to enable live forecasts.")

with tab3:
    st.markdown("#### MLOps Component Configuration")
    
    st.markdown("##### Production Performance Manifest (`data/pipeline_metrics.json`)")
    st.json(latest_metrics)
    
    st.markdown("##### Active Component Processes Execution Console")
    col_exe1, col_exe2 = st.columns(2)
    
    with col_exe1:
        st.markdown("**Data Ingest Execution Log**")
        if st.button("🔌 Execute data_ingestion.py"):
            res = run_pipeline_component("data_ingestion.py")
            st.text_area("Console Log", res.stdout if res.returncode == 0 else res.stderr, height=180, key="ing_log")
            
    with col_exe2:
        st.markdown("**Model Training & Drift Execution Log**")
        if st.button("🧠 Execute model_training.py"):
            res = run_pipeline_component("model_training.py")
            st.text_area("Console Log", res.stdout if res.returncode == 0 else res.stderr, height=180, key="train_log")
