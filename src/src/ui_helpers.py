import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler
from utils import FEATURES

# Directory configuration relative to this file
SRC_SRC_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SRC_SRC_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PYTHON_EXE = sys.executable

def inject_custom_css():
    """Injects custom Obsidian-neon glassmorphism styles."""
    st.markdown("""
    <style>
        /* Theme Adaptability using CSS Variables */
        .stApp {
            background-color: var(--background-color) !important;
            color: var(--text-color) !important;
        }
        
        /* Enforce theme text colors globally */
        .stMarkdown, .stMarkdown p, .stMarkdown span, .stText, p, span, li, label, div[data-testid="stWidgetLabel"] p {
            color: var(--text-color) !important;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: var(--background-color) !important;
            border-right: 1px solid rgba(102, 252, 241, 0.15) !important;
        }
        [data-testid="stSidebar"] * {
            color: var(--text-color) !important;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
            color: var(--primary-color) !important;
        }
        
        /* Header Banner class */
        .header-banner {
            background: linear-gradient(90deg, var(--secondary-background-color) 0%, var(--background-color) 100%) !important;
            padding: 25px;
            border-radius: 14px;
            border: 1px solid rgba(102, 252, 241, 0.15) !important;
            margin-bottom: 25px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }
        .header-desc {
            margin: 5px 0 0 0 !important;
            color: var(--text-color) !important;
            font-size: 1.1rem;
        }
        
        /* Forecast Box class */
        .forecast-box {
            background-color: var(--secondary-background-color) !important;
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }
        .forecast-title {
            margin: 0 !important;
            color: var(--text-color) !important;
        }
        .forecast-confidence {
            margin: 0 !important;
            font-size: 1.1rem;
            color: var(--text-color) !important;
        }
        
        /* Tabs styling */
        button[data-baseweb="tab"] {
            color: var(--text-color) !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--primary-color) !important;
            border-bottom-color: var(--primary-color) !important;
        }
        
        /* Elegant Typography */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-color) !important;
            font-family: 'Outfit', 'Inter', sans-serif;
            font-weight: 700;
            letter-spacing: -0.01em;
        }
        
        /* Neon Glow metrics */
        div[data-testid="stMetricValue"] {
            font-size: 2.3rem !important;
            font-weight: 800 !important;
            color: var(--primary-color) !important;
        }
        
        div[data-testid="metric-container"] {
            background: var(--secondary-background-color) !important;
            border: 1px solid rgba(102, 252, 241, 0.15) !important;
            border-radius: 12px;
            padding: 18px 24px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
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
            background: var(--secondary-background-color) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 12px !important;
            color: var(--text-color) !important;
        }
        
        /* Styled Neon Button */
        .stButton>button {
            background: linear-gradient(135deg, var(--secondary-background-color) 0%, var(--primary-color) 200%) !important;
            color: var(--text-color) !important;
            border: 1px solid var(--primary-color) !important;
            border-radius: 8px !important;
            padding: 12px 28px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            box-shadow: 0 4px 15px rgba(102, 252, 241, 0.1) !important;
        }
        
        .stButton>button:hover {
            background: var(--primary-color) !important;
            color: var(--background-color) !important;
            box-shadow: 0 0 20px rgba(102, 252, 241, 0.6) !important;
            transform: scale(1.02) !important;
        }
    </style>
    """, unsafe_allow_html=True)

def run_pipeline_component(script_name, env_vars=None):
    """Executes a pipeline component script."""
    script_path = os.path.join(SRC_SRC_DIR, script_name)
    cmd = [PYTHON_EXE, script_path]
    
    current_env = os.environ.copy()
    if env_vars:
        current_env.update(env_vars)
        
    with st.spinner(f"Executing pipeline task: {script_name}..."):
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, env=current_env)
    return result

def load_latest_metrics():
    """Loads metrics from the latest daily run."""
    metrics_path = os.path.join(DATA_DIR, "pipeline_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def load_run_history():
    """Loads the history of pipeline runs."""
    history_path = os.path.join(DATA_DIR, "historical_runs.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def load_datasets():
    """Loads the latest ingested market dataset."""
    data_path = os.path.join(DATA_DIR, "latest_ingested_data.csv")
    if os.path.exists(data_path):
        try:
            return pd.read_csv(data_path)
        except Exception:
            pass
    return None

def load_keras_model():
    """Loads the TensorFlow Keras model."""
    model_file = os.path.join(MODELS_DIR, "latest_model.keras")
    if os.path.exists(model_file):
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(model_file)
        except Exception as e:
            st.error(f"Error loading keras model: {e}")
    return None

def plot_price_volume_chart(df_data):
    """Generates the dual-axis Price & Volume Plotly chart."""
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
    return fig

def plot_accuracy_history(run_history):
    """Generates the Validation Accuracy over time Plotly chart."""
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
    return fig2
